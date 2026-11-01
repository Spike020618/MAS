"""Operational schemas. Structural validity never certifies semantic truth."""
from __future__ import annotations
import copy
from experiments.consensus_recommendation.common import fingerprint

RELATIONS = {'support', 'refute', 'unclear'}
OPERATIONS = {'set_answer', 'replace_ranking', 'swap_candidate', 'update_claim', 'abstain'}


def candidate_ids(task):
    return [str(x.get('movie_id', x.get('id'))) if isinstance(x, dict) else str(x)
            for x in task.get('candidates', task.get('candidate_ids', []))]


def validate_decision(task, value):
    if task['kind'] == 'qa':
        options = task.get('options', [])
        if isinstance(options, dict): options = list(options)
        options = [str(x.get('id', x.get('label'))) if isinstance(x, dict) else str(x) for x in options]
        if not isinstance(value, str) or value not in options + ['abstain']:
            raise ValueError('QA decision must be an original option or explicit abstain')
    else:
        if not isinstance(value, list) or not value or any(not isinstance(x, str) for x in value):
            raise ValueError('ranking must be a nonempty list of supplied ID strings')
        if len(value) != len(set(value)) or not set(value) <= set(candidate_ids(task)):
            raise ValueError('ranking contains duplicate or outside candidate IDs')
    return copy.deepcopy(value)


def validate_task(task):
    if not isinstance(task, dict) or not isinstance(task.get('id'), str) or not task['id']:
        raise ValueError('task id required')
    if task.get('kind') not in {'qa', 'rec'} or not isinstance(task.get('prompt'), str):
        raise ValueError('native QA/rec task and prompt required')
    catalog = task.get('source_catalog', task.get('sources', {}))
    if not isinstance(catalog, dict): raise ValueError('source catalog must be ID map')
    for sid, source in catalog.items():
        if not isinstance(source, dict) or source.get('source_id', sid) != sid or not isinstance(source.get('text'), str):
            raise ValueError('source ID/text mismatch')
        scope = source.get('scope', {})
        if not isinstance(scope, dict) or scope.get('task_id') != task['id']:
            raise ValueError('source must be explicitly scoped to current task')
    ids = candidate_ids(task)
    if task['kind'] == 'rec' and (not ids or len(ids) != len(set(ids))):
        raise ValueError('recommendation requires distinct native candidates')
    validate_decision(task, task.get('initial', task.get('decision')))
    return copy.deepcopy(task)


def validate_aeic(report, state):
    """Return a normalized deep copy or raise; never alters state or infers truth."""
    if not isinstance(report, dict) or not {'A', 'E', 'I', 'C'} <= report.keys():
        raise ValueError('AEIC fields required')
    a, e, inference, conclusion = (report[k] for k in ('A', 'E', 'I', 'C'))
    if not isinstance(a, list) or not a or not isinstance(e, list) or not isinstance(inference, list):
        raise ValueError('A nonempty claims, E IDs and I edges must be lists')
    claims = {}
    for claim in a:
        if not isinstance(claim, dict) or not isinstance(claim.get('id'), str) or not claim['id'] or not isinstance(claim.get('text'), str) or not claim['text']:
            raise ValueError('atomic claim id/text required')
        if claim['id'] in claims: raise ValueError('duplicate claim ID in report')
        scope = claim.get('scope')
        if not isinstance(scope, dict) or scope.get('task_id') != state['task_id']:
            raise ValueError('claim scope must match current task')
        if claim['id'] in state['claims'] and state['claims'][claim['id']]['text'] != claim['text']:
            raise ValueError('existing claim cannot silently change; use versioned patch')
        claims[claim['id']] = claim
    if len(e) != len(set(e)) or any(not isinstance(sid, str) or sid not in state['visible_sources'] for sid in e):
        raise ValueError('E must use distinct actually fetched sources')
    available = set(claims) | set(state['claims'])
    for edge in inference:
        if not isinstance(edge, dict) or edge.get('claim_id') not in available or edge.get('source_id') not in e or edge.get('relation') not in RELATIONS:
            raise ValueError('inference must reference a claim and supplied evidence')
    if not isinstance(conclusion, dict) or not isinstance(conclusion.get('proposed_patch'), dict):
        raise ValueError('C.proposed_patch required')
    patch = conclusion['proposed_patch']
    if patch.get('operation') not in OPERATIONS: raise ValueError('unknown patch operation')
    return copy.deepcopy(report)


def scope_allows(source, task_id, purpose='prediction_basis'):
    scope = source.get('scope', {})
    if not isinstance(scope, dict) or scope.get('task_id') != task_id:
        return False
    # Facts about OTHER users and model forecasts can inform a prediction, not
    # certify CURRENT-user preference. No automatic semantic interpretation.
    subject = scope.get('subject', source.get('subject'))
    if purpose == 'current_user_fact' and (subject in {'other_user', 'historical', 'model_forecast'} or source.get('current_user_fact_authority') is False):
        return False
    allowed = source.get('allowed_uses', source.get('allowed_use'))
    if isinstance(allowed, str): allowed = [allowed]
    if allowed is not None and purpose not in allowed:
        if not (purpose == 'prediction_basis' and set(allowed) & {'prediction_basis', 'historical_prediction_basis', 'current_expression', 'catalog_attribute', 'retrieval_basis', 'current_context', 'dialogue_context', 'reference_material'}):
            return False
    return True
