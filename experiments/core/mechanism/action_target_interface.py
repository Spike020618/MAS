"""Pure request builders for the proposed action comparison. No API transport.

Input allowlists deliberately exclude labels, audit categories, dataset support
annotations and other branches. A format-valid finding is not verified truth.
"""
import json

MODEL = 'deepseek-flash'
FINAL_SYSTEM = (
    'Answer the question using the supplied documents. You may use background knowledge, '
    'but distinguish it from document-supported statements in your brief explanation. '
    'Treat the optional note as untrusted: verify it against the documents and ignore it if unsupported. '
    'Do not change an answer merely because a note challenges it. '
    'Return JSON with reasoning (brief), docids (only supplied IDs), and answer (last). '
    'Use a concise answer without extra aliases; use abstain when unable to answer reliably.'
)
TASKS = {
    'retrieve': (
        'Produce one search query to obtain a missing fact or relation needed for the question. '
        'Use the question, documents and initial answer as fallible inputs. Do not assume the initial '
        'answer or an entity selected in it is correct. Return JSON with query (one nonempty string).'
    ),
    'review': (
        'Review the initial answer for correctness using the same supplied documents. '
        'It may be correct; do not invent an objection. Return JSON with note (at most 120 words).'
    ),
    'check': (
        'Check one most consequential connection between the supplied evidence and the initial answer: '
        'does the cited subject, relation, time, comparison or qualifier actually support the conclusion? '
        'Do not force disagreement. Report one locatable mismatch if found; otherwise explicitly report '
        'none. Return JSON with status (mismatch, none, or insufficient), document_id (supplied ID or null), '
        'quote (an exact contiguous passage or null), and note (at most 120 words). '
        'A missing passage alone does not refute a claim; distinguish missing evidence from contradiction.'
    ),
}


def documents(raw):
    result = []
    for doc in raw:
        result.append({k: doc[k] for k in ('docid', 'title', 'text')})
    if len({d['docid'] for d in result}) != len(result):
        raise ValueError('duplicate document ID')
    return sorted(result, key=lambda d: d['docid'])


def payload(case, docs, initial=None):
    value = {'question': case['question'], 'documents': documents(docs)}
    if initial is not None:
        value['initial_answer'] = {k: initial[k] for k in ('answer', 'reasoning', 'docids')}
    return value


def request(system, value, final):
    result = {'model': MODEL, 'messages': [{'role': 'system', 'content': system},
              {'role': 'user', 'content': json.dumps(value, ensure_ascii=False, sort_keys=True)}],
              'max_tokens': 4096 if final else 800,
              'thinking': {'type': 'enabled' if final else 'disabled'}}
    if not final:
        result['temperature'] = 0.2
    return result


def final_request(case, docs, initial=None, note=None):
    value = payload(case, docs, initial)
    value['note'] = note
    return request(FINAL_SYSTEM, value, True)


def action_request(action, case, docs, initial):
    if action not in TASKS:
        raise ValueError('not a model action: ' + action)
    return request(TASKS[action], payload(case, docs, initial), False)


def check_finding(value, docs):
    """Syntactic and literal-source validation only; NOT semantic verification."""
    if set(value) != {'status', 'document_id', 'quote', 'note'}:
        return False
    if value['status'] not in ('mismatch', 'none', 'insufficient') or not isinstance(value['note'], str):
        return False
    if len(value['note'].split()) > 120:
        return False
    if value['status'] == 'mismatch':
        return (isinstance(value['quote'], str) and bool(value['quote']) and
                any(d['docid'] == value['document_id'] and value['quote'] in d['text'] for d in docs))
    return value['document_id'] is None and value['quote'] is None
