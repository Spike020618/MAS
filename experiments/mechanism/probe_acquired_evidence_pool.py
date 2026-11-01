"""Post-hoc whole acquired-evidence strong diagnostic, not a trained method."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from experiments.mechanism import run_open_evidence_screen_v2 as p
from experiments.mechanism import repeat_open_evidence_solver as repeat
from experiments.mechanism.run_upgrade_pilot import CallJournal, freeze, lock, read_rows

ROOT = p.ROOT
OUT = ROOT / 'experiments/results/redesign_v1/acquired_evidence_pool_v1'
DESIGN = ROOT / 'docs/research/redesign_v1/已获取资料全集_诊断协议_2026-10-25.md'


def combine(documents):
    pool = {}; aliases = {}
    for d in sorted(documents, key=lambda d: d['docid']):
        if sha256(d['text'].encode()).hexdigest() != d['text_sha256']: raise ValueError('source hash mismatch')
        key = d['text_sha256']; aliases.setdefault(key, []).append({'docid': d['docid'], 'title': d['title']})
        pool.setdefault(key, {k: d[k] for k in ('docid', 'title', 'text', 'text_sha256')})
    docs = sorted(pool.values(), key=lambda d: d['docid'])
    if sum(len(d['text'].encode()) for d in docs) > 40000: raise ValueError('whole pool over budget; no truncation')
    return docs, aliases


def prepare():
    p.verify(); repeat.verify(); OUT.mkdir(parents=True, exist_ok=True)
    pools = {}; requests = {}
    for c in p.base.read(p.OUT / 'public_cases.json'):
        envs = p.base.read(p.OUT / c['id'] / 'retrieval.json')['environments']
        docs, aliases = combine(c['native_documents'] + [d for env in envs.values() for d in env['documents']])
        pools[c['id']] = {'documents': docs, 'aliases': aliases}
        requests[c['id']] = p.request(c, 'all', docs)
    freeze(OUT / 'pools.json', pools); freeze(OUT / 'requests.json', requests)
    freeze(OUT / 'protocol.json', {'n': 8, 'post_hoc_diagnostic': True, 'maximum_calls': 8,
        'not_gold_selected': True, 'cumulative_usd_proxy_cap': 5,
        'primary_acceptance_sha256': p.base.h(p.OUT / 'final_acceptance.json'),
        'sha256': {str(f.relative_to(ROOT)): p.base.h(f) for f in (Path(__file__), DESIGN, OUT / 'pools.json', OUT / 'requests.json')}})


def verify():
    p.verify(); repeat.verify(); protocol = p.base.read(OUT / 'protocol.json')
    for path, expected in protocol['sha256'].items():
        if p.base.h(ROOT / path) != expected: raise ValueError('frozen pool changed:' + path)
    if p.base.h(p.OUT / 'final_acceptance.json') != protocol['primary_acceptance_sha256']: raise ValueError('primary acceptance changed')


def collect(transport):
    verify()
    requests = p.base.read(OUT / 'requests.json'); pools = p.base.read(OUT / 'pools.json'); outputs = {}
    with lock(OUT):
        journal = CallJournal(OUT, transport, 8)
        responses = read_rows(OUT / 'responses.jsonl')
        if {a['key'] for a in read_rows(OUT / 'attempts.jsonl')} - {r['key'] for r in responses}:
            raise RuntimeError('unresolved attempt; no automatic retry')
        for cid, req in requests.items():
            if cid not in {r['key'] for r in responses}:
                paths = list(p.OUT.rglob('responses.jsonl')) + [repeat.OUT / 'responses.jsonl', OUT / 'responses.jsonl']
                raws = [r for path in paths for r in read_rows(path)]
                spent = sum((json.loads(r['response']['body'])['usage']['prompt_tokens'] * .3 +
                    json.loads(r['response']['body'])['usage']['completion_tokens'] * 1.2) / 1e6 for r in raws)
                reserve = ((len(p.base.encoded(req)) + 4096) * .3 + req['max_tokens'] * 1.2) / 1e6
                if spent + reserve > 5: raise RuntimeError('hour cumulative proxy exhausted')
            row = journal.call(cid, req); outputs[cid] = p.parse(row, 'all', pools[cid]['documents'])
            print('whole pool sealed', len(outputs), '/8', flush=True)
        freeze(OUT / 'sealed_outputs.json', outputs)


def analyze():
    verify(); outputs = p.base.read(OUT / 'sealed_outputs.json'); labels = p.base.read(p.OUT / 'private_labels.json')
    first = p.base.read(p.OUT / 'sealed_outputs.json')
    if set(outputs) != set(labels): raise ValueError('incomplete pool experiment; no selective scores')
    scores = {c: p.answer_score(outputs[c]['answer'], labels[c]) for c in labels}
    comparisons = {}
    for arm in ('direct', 'native', 'all', 'sequential'):
        comparisons[arm] = {'corrected': [c for c in labels if scores[c]['em'] and not p.answer_score(first[c][arm]['answer'], labels[c])['em']],
            'harmed': [c for c in labels if not scores[c]['em'] and p.answer_score(first[c][arm]['answer'], labels[c])['em']]}
    raw = read_rows(OUT / 'responses.jsonl')
    result = {'scope': 'post-hoc whole acquired pool, not independent test or game gain', 'n': 8,
        'em_correct': sum(s['em'] for s in scores.values()), 'mean_f1': sum(s['f1'] for s in scores.values()) / 8,
        'invalid': sum(not s['valid'] for s in outputs.values()), 'comparisons': comparisons,
        'physical_calls': len(raw), 'physical_tokens': sum(json.loads(r['response']['body'])['usage']['total_tokens'] for r in raw),
        'sum_http_seconds': sum(r['wall_seconds'] for r in raw),
        'usd_proxy': sum((json.loads(r['response']['body'])['usage']['prompt_tokens'] * .3 +
            json.loads(r['response']['body'])['usage']['completion_tokens'] * 1.2) / 1e6 for r in raw)}
    freeze(OUT / 'result.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('prepare', 'run', 'analyze', 'replay'))
    stage = parser.parse_args().stage
    if stage == 'prepare': prepare()
    elif stage == 'analyze': analyze()
    elif stage == 'replay':
        def blocked(req): raise AssertionError('unexpected replay API')
        collect(blocked); analyze()
    else:
        from experiments.mechanism.study_explicit_thinking_control import transport
        collect(transport); analyze()
