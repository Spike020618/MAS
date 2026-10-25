"""Standard calibrated reliability readout, not a novel or optimal mechanism.

The log-odds rule assumes symmetric wrong-choice likelihood as a heuristic.
Actual report dependence and calibration transfer are NOT guaranteed.
"""
from collections import Counter
from math import log


def calibrate(rows, backends):
    """Offline calibration only. Invalid/abstaining reports count as incorrect."""
    if not rows or len(set(backends)) != len(backends):
        raise ValueError('nonempty calibration and unique backends required')
    counts = {b: [0, 0] for b in backends}
    question_ids = set()
    for row in rows:
        if row['id'] in question_ids:
            raise ValueError('duplicate calibration question')
        question_ids.add(row['id'])
        if row['gold'] not in row['choices']:
            raise ValueError('invalid calibration label')
        for report in row['reports']:
            if report['backend'] not in counts:
                raise ValueError('unknown backend')
            cell = counts[report['backend']]
            cell[0] += report['answer'] == row['gold']
            cell[1] += 1
    if any(n == 0 for _, n in counts.values()):
        raise ValueError('missing backend calibration')
    return {'rule': 'beta1-smoothed accuracy; log((K-1)*p/(1-p)) vote weight',
            'calibration_questions': len(question_ids),
            'counts': {b: {'correct': c, 'reports': n} for b, (c, n) in counts.items()},
            'probabilities': {b: (c+1)/(n+2) for b, (c, n) in counts.items()},
            'limits': 'repeated reports are correlated; probabilities not certified or task-conditionally calibrated'}


def decide(reports, choices, model=None):
    """Inference accepts no gold label, question text, or domain features."""
    if len(set(choices)) != len(choices) or len(choices) < 2 or 'abstain' in choices:
        raise ValueError('invalid choices')
    scores = {c: 0. for c in choices}
    for r in reports:
        if r['answer'] not in choices:
            continue
        if model is None:
            w = 1.
        else:
            if r['backend'] not in model['probabilities']:
                raise ValueError('uncalibrated backend')
            p = model['probabilities'][r['backend']]
            if not 0 < p < 1:
                raise ValueError('invalid calibrated probability')
            w = log((len(choices)-1)*p/(1-p))
        scores[r['answer']] += w
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    # Zero evidence, exact ties, and a nonpositive winning support abstain.
    answer = ranked[0][0] if ranked[0][1] > 1e-12 and ranked[0][1]-ranked[1][1] > 1e-12 else 'abstain'
    return {'answer': answer, 'scores': scores, 'weighted': model is not None}
