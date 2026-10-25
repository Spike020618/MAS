"""Experimental premise review interface. No network access or task-specific rules.

This is not a proven mechanism, factual verifier, or equilibrium implementation.
"""
import json
from collections import Counter

ARMS = ('ordinary', 'premise')
COMMON = (
    'Solve the original question carefully. The original wording and answer criterion take priority. '
    'The sealed answers below are fallible model outputs, not evidence of truth. '
    'You may keep, revise or abstain. No external source or tool has been accessed; '
    'do not claim verification or treat agreement as proof. Give sufficient reasoning within the limit. '
    'Return JSON with answer (one registered choice or abstain) and reasoning (text). '
)
ORDINARY = 'Review the supplied answers and reasoning freely, and reach your own conclusion.'
PREMISE = (
    'Focus on a premise or inference step that could change the answer, if one exists. '
    'Distinguish an original question condition from an added assumption. '
    'Check whether the conclusion follows without changing the question scope or answer criterion. '
    'If a premise remains unresolved, say so; do not invent a dispute or a new fact. '
    'Explain any change through the relevant premise. No fixed decomposition or four-field format is required.'
)


def request(question, choices, sealed_reports, arm, model, max_tokens=800):
    """Accept input-only reports; labels cannot be forwarded through extra fields."""
    if arm not in ARMS or not isinstance(question, str) or not question.strip():
        raise ValueError('invalid arm/question')
    if len(choices) < 2 or len(set(choices)) != len(choices) or 'abstain' in choices:
        raise ValueError('invalid choices')
    reports = []
    for i, report in enumerate(sealed_reports):
        if set(report) != {'answer', 'reasoning'} or report['answer'] not in (*choices, 'abstain') or not isinstance(report['reasoning'], str):
            raise ValueError('reports must contain only answer and reasoning')
        reports.append({'ref': f'initial:{i}', **report})
    if not reports or not model or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('missing reports/model or invalid budget')
    return {'model': model, 'temperature': .2, 'max_tokens': max_tokens,
            'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': COMMON + (ORDINARY if arm == 'ordinary' else PREMISE)},
                         {'role': 'user', 'content': json.dumps({'question': question, 'choices': choices,
                                                               'sealed_reports': reports}, ensure_ascii=False, sort_keys=True)}]}


def decode(raw, choices):
    """Exact answer field, never scan reasoning for the last choice word."""
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or obj.get('answer') not in (*choices, 'abstain') or not isinstance(obj.get('reasoning'), str):
            raise ValueError('invalid answer/reasoning')
        return {'answer': obj['answer'], 'reasoning': obj['reasoning'], 'valid': True}
    except (ValueError, TypeError):
        return {'answer': None, 'reasoning': None, 'valid': False}


def paired_summary(rows):
    """Offline scorer only; rows must include ALL prespecified cases, no filtering.

    initial is a single sealed panel decision. Each review arm has one call.
    Costs include the common initial panel in each arm, not as free input.
    """
    keys = [r['id'] for r in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError('nonempty unique paired cases required')
    counts, tokens, calls = Counter(), Counter(), Counter()
    for row in rows:
        if set(row['reviews']) != set(ARMS):
            raise ValueError('missing review arm; do not silently discard failures')
        gold = row['gold']
        if gold not in row['choices']:
            raise ValueError('invalid offline label')
        initial_correct = row['initial_answer'] == gold
        counts['initial_correct'] += initial_correct
        for arm in ARMS:
            r = row['reviews'][arm]
            obs = decode(r['raw'], row['choices'])
            correct = obs['answer'] == gold
            counts[arm + '_correct'] += correct
            counts[arm + '_invalid'] += not obs['valid']
            counts[arm + '_corrections'] += correct and not initial_correct
            counts[arm + '_harms'] += initial_correct and not correct
            for field in ('tokens', 'calls'):
                values = (row['initial_' + field], r[field])
                if any(type(v) is not int or v < 0 for v in values):
                    raise ValueError('unknown costs must not be encoded as zero or estimated')
                (tokens if field == 'tokens' else calls)[arm] += sum(values)
        oc = decode(row['reviews']['ordinary']['raw'], row['choices'])['answer'] == gold
        pc = decode(row['reviews']['premise']['raw'], row['choices'])['answer'] == gold
        counts['premise_vs_ordinary_corrections'] += pc and not oc
        counts['premise_vs_ordinary_harms'] += oc and not pc
    return {'n': len(rows), 'counts': dict(counts), 'total_tokens_with_initial': dict(tokens),
            'total_calls_with_initial': dict(calls),
            'scope': 'paired descriptive results, not factual process validation or proof of benefit'}
