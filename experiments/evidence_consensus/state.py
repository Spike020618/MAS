"""Versioned shared state and non-destructive, source-bound local updates."""
from __future__ import annotations
import copy
from experiments.consensus_recommendation.common import fingerprint
from .schema import validate_task, validate_decision, validate_aeic, scope_allows


def new_state(task):
    task = validate_task(task)
    catalog = copy.deepcopy(task.get('source_catalog', task.get('sources', {})))
    return {'task_id': task['id'], 'kind': task['kind'], 'task': task,
            'version': 0, 'decision': copy.deepcopy(task.get('initial', task.get('decision'))),
            'claims': {}, 'catalog': catalog, 'catalog_hash': fingerprint(catalog),
            'visible_sources': {}, 'events': []}


def check_integrity(state):
    if fingerprint(state['catalog']) != state['catalog_hash']:
        raise ValueError('immutable source catalog altered')
    for sid, source in state['visible_sources'].items():
        if sid not in state['catalog'] or source != state['catalog'][sid]:
            raise ValueError('fetched source altered or not in catalog')


def initialize_decision(state, value):
    """Explicit first-generation initialization; not a claimed evidence correction."""
    check_integrity(state)
    if state['version'] != 0 or any(e.get('kind') == 'initialize_decision' for e in state['events']):
        raise ValueError('initial decision may be committed once, before any patch')
    result = copy.deepcopy(state)
    result['decision'] = validate_decision(state['task'], value)
    result['events'].append({'kind': 'initialize_decision', 'decision_hash': fingerprint(value), 'semantic_truth_certified': False})
    return result


def register_report(state, report):
    check_integrity(state)
    report = validate_aeic(report, state)
    result = copy.deepcopy(state)
    for claim in report['A']:
        if claim['id'] not in result['claims']:
            result['claims'][claim['id']] = {**copy.deepcopy(claim), 'version': 0, 'status': 'unverified'}
    result['events'].append({'kind': 'register_report', 'report_hash': fingerprint(report),
                             'claims': [x['id'] for x in report['A']], 'semantic_truth_certified': False})
    return result


def apply_patch(state, patch):
    """Return {accepted, reason, state}; rejection preserves exact original state.

    A successful update means operationally valid, never semantic certification.
    """
    try:
        check_integrity(state)
        if not isinstance(patch, dict): raise ValueError('patch must be a dictionary')
        if type(patch.get('expected_version')) is not int or patch['expected_version'] != state['version']:
            raise ValueError('stale or missing expected_version')
        operation = patch.get('operation')
        if operation == 'abstain':
            return {'accepted': False, 'reason': 'explicit_abstention', 'state': copy.deepcopy(state), 'semantic_truth_certified': False}
        evidence = patch.get('evidence_ids', [])
        if not isinstance(evidence, list) or not evidence or len(evidence) != len(set(evidence)):
            raise ValueError('non-abstain patch requires distinct evidence IDs')
        purpose = patch.get('evidence_purpose', 'prediction_basis')
        for sid in evidence:
            if sid not in state['visible_sources'] or not scope_allows(state['visible_sources'][sid], state['task_id'], purpose):
                raise ValueError('evidence not fetched or outside source authority')
        preserve = patch.get('preserve_ids', [])
        if not isinstance(preserve, list) or any(cid not in state['claims'] for cid in preserve):
            raise ValueError('preserve_ids must identify existing claims')
        result = copy.deepcopy(state)
        if operation == 'set_answer':
            if state['kind'] != 'qa': raise ValueError('answer operation used for non-QA task')
            result['decision'] = validate_decision(state['task'], patch.get('value'))
        elif operation == 'replace_ranking':
            if not state['task'].get('metadata', {}).get('allow_full_rewrite_ablation', False):
                raise ValueError('full ranking rewrite forbidden; use a scoped candidate swap')
            if state['kind'] != 'rec': raise ValueError('ranking operation used for non-rec task')
            value = validate_decision(state['task'], patch.get('value'))
            # Permit a Top10 view of the existing full candidate ranking. The
            # remainder is kept in its exact prior order and candidates stay closed.
            result['decision'] = value + [x for x in state['decision'] if x not in value]
        elif operation == 'swap_candidate':
            if state['kind'] != 'rec': raise ValueError('swap operation used for non-rec task')
            a, b = patch.get('a'), patch.get('b')
            if a == b or a not in state['decision'] or b not in state['decision']:
                raise ValueError('swap must name two distinct existing native IDs')
            i, j = result['decision'].index(a), result['decision'].index(b)
            result['decision'][i], result['decision'][j] = b, a
        elif operation == 'update_claim':
            cid = patch.get('target_id')
            if cid not in state['claims'] or cid in preserve: raise ValueError('claim target missing or explicitly preserved')
            if patch.get('claim_version') != state['claims'][cid]['version']: raise ValueError('stale claim version')
            value = patch.get('value')
            if not isinstance(value, str) or not value: raise ValueError('replacement claim text required')
            result['claims'][cid]['text'] = value
            result['claims'][cid]['version'] += 1
            result['claims'][cid]['status'] = 'unverified'
        else: raise ValueError('unknown patch operation')
        if any(result['claims'][cid] != state['claims'][cid] for cid in preserve):
            raise ValueError('preserved claim changed')
        if result['decision'] == state['decision'] and result['claims'] == state['claims']:
            return {'accepted': False, 'reason': 'no_change', 'state': copy.deepcopy(state), 'semantic_truth_certified': False}
        result['version'] += 1
        result['events'].append({'kind': 'patch', 'patch': copy.deepcopy(patch), 'before_version': state['version'],
                                  'after_version': result['version'], 'semantic_truth_certified': False})
        return {'accepted': True, 'reason': 'operationally_valid', 'state': result, 'semantic_truth_certified': False}
    except (ValueError, KeyError, TypeError) as exc:
        return {'accepted': False, 'reason': str(exc), 'state': copy.deepcopy(state), 'semantic_truth_certified': False}
