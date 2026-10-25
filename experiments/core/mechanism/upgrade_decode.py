"""P2-v2 failure policy: preserve raw failures, explicit non-LLM abstention.

Never fills missing AEIC reasoning or salvages C selectively. HTTP, truncation
and usage problems remain hard stops; completed invalid reports become failed
slots, not successful AEIC. No answer key involved.
"""
import json
from .upgrade_game import validate_report

INVALID_MARKER = '[INVALID_AEIC_REPORT_NOT_MODEL_OUTPUT]'


def is_failure(report):
    return report['A'].startswith(INVALID_MARKER)


def parse(row, choices):
    response = row['response']
    if response['http_status'] != 200:
        raise ValueError('HTTP failure retained; manual transport reconciliation required')
    body = json.loads(response['body'])
    choice = body['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('unfinished generation retained; no automatic repair')
    usage = {k: body.get('usage', {}).get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    if any(type(v) is not int or v < 0 for v in usage.values()) or usage['prompt_tokens'] + usage['completion_tokens'] != usage['total_tokens']:
        raise ValueError('invalid provider usage')
    if not body.get('model'):
        raise ValueError('returned model missing')
    try:
        report = json.loads(choice['message']['content'])
        validate_report(report, choices)
        if is_failure(report):
            raise ValueError('reserved executor failure marker')
    except (ValueError, TypeError, KeyError):
        report = {'A': INVALID_MARKER + ' completed response failed the registered AEIC schema.',
                  'E': [], 'I': 'Executor failure placeholder, not model inference; see raw response.', 'C': 'abstain'}
    return report, usage
