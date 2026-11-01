"""Unrun contract repair. Does not replace frozen v1 or change old scores.

No task/answer-specific logic. Literal citation validation remains nonsemantic.
"""
from experiments.core.mechanism import action_target_interface as v1


def normalize_report(kind, value, docs):
    if not isinstance(value, dict):
        return None
    if kind == 'review':
        text = value.get('note')
        if isinstance(text, str) and 0 < len(text.split()) <= 120:
            # Extra model self-scores never enter downstream solver inputs.
            return {'note': text}
        return None
    if kind != 'check':
        raise ValueError('only review/check contract repair')
    keys = ('status', 'document_id', 'quote', 'note')
    if not all(k in value for k in keys):
        return None
    result = {k: value[k] for k in keys}
    if result['status'] not in ('none', 'mismatch', 'insufficient'):
        return None
    if not isinstance(result['note'], str) or not 0 < len(result['note'].split()) <= 120:
        return None
    key, quote = result['document_id'], result['quote']
    if key is None and quote is None:
        return result if result['status'] != 'mismatch' else None
    if not isinstance(key, str) or not isinstance(quote, str) or not quote:
        return None
    if not any(d['docid'] == key and quote in d['text'] for d in docs):
        return None
    return result


def action_request(kind, case, docs, initial):
    req = v1.action_request(kind, case, docs, initial)
    req['messages'][0]['content'] += ' Return only the requested JSON keys; do not add correctness scores.'
    if kind == 'check':
        req['messages'][0]['content'] += (
            ' For none or insufficient, you may cite a real supplied passage, or set both document_id '
            'and quote to null. For mismatch, a registered document_id and exact nonempty quote are required.'
        )
    return req


def final_request(case, docs, initial=None, note=None):
    req = v1.final_request(case, docs, initial, note)
    req['messages'][0]['content'] += (
        ' Put explanations only in reasoning. For a yes/no question, answer must be exactly yes or no '
        '(or abstain if unknown). Do not append parentheses, acronyms or explanatory sentences to answer.'
    )
    return req
