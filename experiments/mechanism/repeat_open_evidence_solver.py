"""Prospective repeat of frozen pair requests, never a new independent test set."""
import argparse
import json
from pathlib import Path
from experiments.mechanism import run_open_evidence_screen_v2 as primary
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows, digest

ROOT = primary.ROOT
OUT = ROOT / 'experiments/results/redesign_v1/evaluator_replication_v1'
DESIGN = ROOT / 'docs/research/redesign_v1/求解器复评_预定补充协议_2026-10-25.md'
ARMS = ('p01', 'p02', 'p12')


def original(case, arm):
    folder = primary.OUT / case['id']
    rows = read_rows(folder / 'responses.jsonl')
    by_key = {r['key']: r for r in rows}
    key = arm if arm in by_key else primary.base.read(folder / (arm + '_alias.json'))['source_key']
    row = by_key[key]
    if digest(row['request']) != row['request_hash'] or digest(row['response']) != row['response_hash']:
        raise ValueError('source request/response altered')
    payload = json.loads(row['request']['messages'][1]['content'])
    if set(payload) != {'question', 'documents'} or payload['question'] != case['question']:
        raise ValueError('not a registered evidence request')
    return row['request'], payload['documents']


def prepare():
    primary.verify()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (primary.OUT / 'final_acceptance.json').exists(): raise ValueError('primary not accepted')
    cases = primary.base.read(primary.OUT / 'public_cases.json')
    requests = {c['id']: {a: original(c, a)[0] for a in ARMS} for c in cases}
    freeze(OUT / 'requests.json', requests)
    freeze(OUT / 'protocol.json', {'version': 'repeat-solver-v1', 'n': 8, 'maximum_calls': 24,
        'primary_acceptance_sha256': primary.base.h(primary.OUT / 'final_acceptance.json'),
        'requests_sha256': primary.base.h(OUT / 'requests.json'), 'design_sha256': primary.base.h(DESIGN),
        'code_sha256': primary.base.h(Path(__file__)), 'cumulative_usd_proxy_cap': 5,
        'no_new_questions': True, 'no_policy_fit': True})


def verify():
    primary.verify(); p = primary.base.read(OUT / 'protocol.json')
    for path, field in ((Path(__file__), 'code_sha256'), (DESIGN, 'design_sha256'),
        (OUT / 'requests.json', 'requests_sha256'), (primary.OUT / 'final_acceptance.json', 'primary_acceptance_sha256')):
        if primary.base.h(path) != p[field]: raise ValueError('frozen input changed: ' + str(path))
    return p


def collect(transport):
    verify(); requests = primary.base.read(OUT / 'requests.json'); outputs = {}
    with lock(OUT):
        journal = CallJournal(OUT, transport, 24)
        for attempt in read_rows(OUT / 'attempts.jsonl'):
            if attempt['key'] not in {r['key'] for r in read_rows(OUT / 'responses.jsonl')}:
                raise RuntimeError('unresolved attempt; do not retry')
        for cid, arms in requests.items():
            outputs[cid] = {}
            for arm, req in arms.items():
                existing = read_rows(OUT / 'responses.jsonl')
                match = next((r for r in existing if r['request'] == req), None)
                if match is not None:
                    match = journal.call(match['key'], req)
                if match is None:
                    all_raw = existing + [r for p in primary.OUT.rglob('responses.jsonl') for r in read_rows(p)]
                    spent = sum((json.loads(r['response']['body'])['usage']['prompt_tokens'] * .3 +
                        json.loads(r['response']['body'])['usage']['completion_tokens'] * 1.2) / 1e6 for r in all_raw)
                    reserve = ((len(primary.base.encoded(req)) + 4096) * .3 + req['max_tokens'] * 1.2) / 1e6
                    if spent + reserve > 5: raise RuntimeError('cumulative budget exhausted')
                    match = journal.call(cid + '/' + arm, req)
                docs = json.loads(req['messages'][1]['content'])['documents']
                outputs[cid][arm] = primary.parse(match, arm, docs)
            print('repeat sealed', len(outputs), '/8', flush=True)
        freeze(OUT / 'sealed_outputs.json', outputs)
    return outputs


def analyze():
    verify(); outputs = primary.base.read(OUT / 'sealed_outputs.json')
    first = primary.base.read(primary.OUT / 'sealed_outputs.json')
    labels = primary.base.read(primary.OUT / 'private_labels.json')
    if set(outputs) != set(labels) or any(set(v) != set(ARMS) for v in outputs.values()):
        raise ValueError('incomplete repeat; do not score selectively')
    metrics = {}; changes = []
    for arm in ARMS:
        scores = [primary.answer_score(outputs[c][arm]['answer'], labels[c]) for c in labels]
        metrics[arm] = {'em_correct': sum(s['em'] for s in scores), 'mean_f1': sum(s['f1'] for s in scores) / 8,
            'invalid': sum(not outputs[c][arm]['valid'] for c in labels)}
        for c in labels:
            old = primary.answer_score(first[c][arm]['answer'], labels[c]); new = primary.answer_score(outputs[c][arm]['answer'], labels[c])
            if old != new: changes.append({'id': c, 'arm': arm, 'first': old, 'repeat': new})
    raw = read_rows(OUT / 'responses.jsonl')
    result = {'n_independent_questions': 8, 'repeat_not_new_sample': True, 'metrics': metrics,
        'score_changes': changes, 'physical_calls': len(raw),
        'physical_tokens': sum(json.loads(r['response']['body'])['usage']['total_tokens'] for r in raw),
        'sum_http_seconds': sum(r['wall_seconds'] for r in raw),
        'usd_proxy': sum((json.loads(r['response']['body'])['usage']['prompt_tokens'] * .3 +
            json.loads(r['response']['body'])['usage']['completion_tokens'] * 1.2) / 1e6 for r in raw),
        'pair_oracle_correct_not_deployable': sum(max(primary.answer_score(outputs[c][a]['answer'], labels[c])['em'] for a in ARMS) for c in labels)}
    freeze(OUT / 'result.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'run', 'analyze', 'replay'))
    stage = p.parse_args().stage
    if stage == 'prepare': prepare()
    elif stage == 'analyze': analyze()
    elif stage == 'replay':
        def blocked(req): raise AssertionError('replay attempted API')
        collect(blocked); analyze()
    else:
        from experiments.mechanism.study_explicit_thinking_control import transport
        collect(transport); analyze()
