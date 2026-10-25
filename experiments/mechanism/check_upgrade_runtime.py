"""Persist a clearly synthetic 12-case dry run and independent audit (zero API)."""
import argparse
import json
from pathlib import Path

from experiments.mechanism.run_upgrade_pilot import freeze, run
from experiments.mechanism.audit_upgrade_pilot import audit


def mock(request):
    report = {'A': 'engineering fixture only', 'E': [], 'I': 'no factual research inference', 'C': 'abstain'}
    body = {'model': 'OFFLINE-MOCK-NOT-DEEPSEEK', 'id': 'mock-not-provider-receipt',
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(report)}}],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
    return {'http_status': 200, 'body': json.dumps(body), 'requested_model': 'OFFLINE-MOCK'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = [{'id': 'OFFLINE-FIXTURE-' + str(i), 'source_id': 'synthetic-not-dataset-' + str(i),
              'question': 'Engineering fixture ' + str(i) + ': no factual answer intended.',
              'choices': ['yes', 'no']} for i in range(12)]
    first = run(cases, args.output, mock, {'mode': 'offline-mock-zero-api'})

    def must_not_call(request):
        raise AssertionError('resume attempted an extra transport call')

    second = run(cases, args.output, must_not_call, {'mode': 'offline-mock-zero-api'})
    assert first == second
    result = {**audit(args.output), 'api_calls': 0, 'mock_responses': 72,
              'resume_added_calls': 0, 'scope': 'synthetic offline runtime only, NOT the 12-question research pilot'}
    freeze(args.output / 'offline_audit.json', result)
    print(json.dumps(result))
