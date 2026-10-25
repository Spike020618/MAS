"""Predeclared low-dimensional proxy, trained ONLY on calibration outcomes."""
from collections import defaultdict
from itertools import product

from .upgrade_game import Contract, aggregate, digest, validate_report

PROFILES = tuple(product((0, 1), repeat=3))
COST_GRID = (.005, .01, .02, .05)


def state_key(reports):
    if len(reports) != 3:
        raise ValueError('three registered agents required')
    # Restricted-growth coding preserves agent positions, not answer identities.
    # A/E/I remain report fields, NOT independently verified predictive signals.
    seen, codes = {}, []
    for report in reports:
        vote = report['C']
        if vote == 'abstain':
            codes.append('_')
        else:
            if vote not in seen:
                seen[vote] = str(len(seen))
            codes.append(seen[vote])
    return ''.join(codes)


def outcomes(initial, upgraded, choices, label):
    if len(initial) != 3 or len(upgraded) != 3 or label not in choices:
        raise ValueError('complete potential reports and registered gold required')
    for report in initial + upgraded:
        validate_report(report, choices)
    return [int(aggregate([upgraded[i] if a else initial[i] for i, a in enumerate(p)], choices) == label)
            for p in PROFILES]


def fit(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('unique calibration questions required')
    grouped = defaultdict(list)
    for row in rows:
        if len(row['scores']) != 8 or any(s not in (0, 1) for s in row['scores']):
            raise ValueError('complete objective outcomes required')
        grouped[row['state']].append(row['scores'])
    # Jeffreys-smoothed pooled estimate, then four pseudo-question shrinkage.
    pooled = [(sum(r['scores'][j] for r in rows) + .5) / (len(rows) + 1) for j in range(8)]
    states = {key: {'n': len(values), 'values': [
        (sum(v[j] for v in values) + 4 * pooled[j]) / (len(values) + 4) for j in range(8)]}
        for key, values in sorted(grouped.items())}
    return {'version': 'upgrade-proxy-mean-v1', 'calibration_ids_hash': digest(sorted(r['id'] for r in rows)),
            'n': len(rows), 'pooled': pooled, 'states': states, 'shrinkage': 4,
            'features': 'initial C equality/abstention pattern only; no text or dataset',
            'fit_scope': 'calibration only, not an accuracy guarantee'}


def values(model, initial):
    key = state_key(initial)
    return tuple(model['states'].get(key, {'values': model['pooled']})['values'])


def contract_for(model, initial, cost=.01):
    if cost not in COST_GRID:
        raise ValueError('cost outside predeclared grid')
    beta = 1 / 6
    return Contract(values(model, initial), ((0, beta * cost),) * 3, 1,
                    'calibration-proxy:' + digest(model))
