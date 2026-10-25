"""Transport-mocked request-boundary audit; never calls real model clients.

Current code is tested, then key prompt ASTs compared with archived source.
This is not a reconstruction of unavailable historical raw request bodies.
"""
import ast
from hashlib import sha256
import json
import subprocess
from unittest.mock import patch

from experiments.mechanism.attribute_legacy_gain import ROOT, OUT
from experiments.mechanism.collect import generate_one


def capture(action, contract, reputation):
    requests = []
    def fake(client, messages, temperature, max_tokens, stage):
        requests.append({'messages': messages, 'temperature': temperature,
                         'max_tokens': max_tokens, 'stage': stage, 'backend': client})
        raw = json.dumps({'ACTION': action}) if stage == 'choose' else json.dumps(
            {'A': 'premise', 'E': ['evidence'], 'I': 'inference', 'C': 'Yes'})
        return raw, {'stage': stage, 'prompt_tokens': 1, 'completion_tokens': 1,
                     'latency_s': 0, 'status': 200, 'backend': client, 'model': client}
    with patch('experiments.mechanism.collect.client_for', side_effect=lambda backend: backend), \
         patch('experiments.mechanism.collect._call_json', side_effect=fake):
        generate_one('Fixed public question?', 0, 'deepseek', contract,
            'No peer information is visible. Reason independently.', reputation,
            False, resource_tiered=True)
    return {'choice': requests[0], 'execution': requests[1:]}


def analyze():
    checks = []
    for action in ['low_effort', 'aeic_effort', 'verify', 'defer_imitate']:
        a = capture(action, 'No credit contract A', 0.)
        b = capture(action, 'Different credit contract B', -19.)
        checks.append({'action': action, 'same_execution_requests': a['execution'] == b['execution'],
                       'choice_request_changes': a['choice'] != b['choice'],
                       'execution_request': a['execution']})
    rel = 'experiments/mechanism/collect.py'
    path = ROOT / rel
    historical = subprocess.run(['git','show','0f94c09:'+rel], cwd=ROOT, text=True,
                                capture_output=True, check=True).stdout
    old, current = ast.parse(historical), ast.parse(path.read_text())
    comparable = {}
    for name in ['execution_prompt','peer_context','build_choice_prompt','generate_one','run_record']:
        a = next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b = next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
        comparable[name] = ast.dump(a) == ast.dump(b)
    return {'no_api_calls': True, 'checks': checks, 'archived_commit': '0f94c09',
        'prompt_function_AST_matches_archived_commit': comparable,
        'current_source_sha256': sha256(path.read_bytes()).hexdigest(),
        'archived_source_sha256': sha256(historical.encode()).hexdigest(),
        'interpretation': 'in this implementation contract/history change choice, not the report request conditional on action; action bundles backend/cap/instruction/self-check',
        'limits': ['mocked current execution and archived AST comparison, not historical raw request attestation',
                   'generation/record bodies changed since archived commit; exact historical execution not independently reconstructed',
                   'request equivalence does not force equal stochastic outputs or eliminate time/backend changes']}


def main():
    r = analyze()
    if not all(x['same_execution_requests'] and x['choice_request_changes'] for x in r['checks']):
        raise ValueError('contract unexpectedly enters conditional execution request')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'prompt_boundary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='checks'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
