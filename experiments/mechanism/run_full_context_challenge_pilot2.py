"""User-authorized first two frozen sources, at most90 calls/$5 proxy cap."""
import argparse
import json
from pathlib import Path
import time

from experiments.mechanism import run_full_context_challenge as parent
from experiments.mechanism.run_upgrade_pilot import append, freeze, lock, read_rows

OUT = parent.OUT / 'pilot2'


def prepare():
    parent.verify(parent.OUT)
    with lock(OUT):
        cases = parent.read(parent.OUT / 'public_cases.json')[:2]
        labels = parent.read(parent.OUT / 'private_labels.json')
        freeze(OUT / 'public_cases.json', cases)
        freeze(OUT / 'private_labels.json', {c['id']: labels[c['id']] for c in cases})
        freeze(OUT / 'authorization.json', {
            'human_request': '小批量启动吧', 'groups': 2, 'max_calls': 90,
            'max_usd_proxy': 5, 'automatic_expansion': False,
            'parent_protocol_sha256': parent.h(parent.OUT / 'protocol.json'),
            'runner_sha256': parent.h(Path(__file__)),
            'public_cases_sha256': parent.h(OUT / 'public_cases.json'),
            'private_labels_sha256': parent.h(OUT / 'private_labels.json'),
            'price_checked_date': '2026-10-25',
            'selection': 'first two of previously frozen input-hash order; no label-based selection'})


def run():
    parent.verify(parent.OUT)
    a = parent.read(OUT / 'authorization.json')
    assert a['runner_sha256'] == parent.h(Path(__file__))
    assert a['parent_protocol_sha256'] == parent.h(parent.OUT / 'protocol.json')
    assert a['public_cases_sha256'] == parent.h(OUT / 'public_cases.json')
    assert a['private_labels_sha256'] == parent.h(OUT / 'private_labels.json')
    assert time.strftime('%Y-%m-%d', time.gmtime()) == a['price_checked_date']
    cases = parent.read(OUT / 'public_cases.json')
    assert len(cases) == 2
    from experiments.core.llm import LLM_BACKENDS
    from urllib.parse import urlparse
    assert urlparse(LLM_BACKENDS['deepseek']['base_url']).hostname == 'api.deepseek.com'
    assert LLM_BACKENDS['deepseek']['api_key']
    from experiments.mechanism.study_explicit_thinking_control import transport
    with lock(OUT):
        responses = [r for p in OUT.rglob('responses.jsonl') for r in read_rows(p)]
        attempts = [r for p in OUT.rglob('attempts.jsonl') for r in read_rows(p)]
        assert len(attempts) <= 90
        for p in OUT.rglob('attempts.jsonl'):
            if {r['key'] for r in read_rows(p)} - {r['key'] for r in read_rows(p.with_name('responses.jsonl'))}:
                raise RuntimeError('unresolved attempt; no retry')
        used = 0
        for r in responses:
            usage = json.loads(r['response']['body'])['usage']
            used += (usage['prompt_tokens'] * .3 + usage['completion_tokens'] * 1.2) / 1e6
        spend = {'used': used, 'cap': 5}
        outputs = {}; start = time.perf_counter()
        try:
            for c in cases:
                outputs[c['id']] = parent.collect(c, OUT, transport, spend)
                print('sealed source', c['id'], flush=True)
            freeze(OUT / 'sealed_outputs.json', outputs)
            freeze(OUT / 'sealed_readouts.json', {cid: {k: {m: s[m]['verdict'] for m in (
                'ordinary0', 'ordinary2', 'fixed_judge', 'challenge_judge', 'nonadaptive', 'interactive')}
                for k, s in r['claims'].items()} for cid, r in outputs.items()})
            parent.analyze(OUT)
            print(json.dumps(parent.read(OUT / 'result.json')['metrics'], ensure_ascii=False), flush=True)
        finally:
            append(OUT / 'invocations.jsonl', {'elapsed_seconds': time.perf_counter() - start,
                'completed_source_groups': len(outputs), 'known_cost_proxy_usd': spend['used']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'run'))
    if p.parse_args().stage == 'prepare': prepare()
    else: run()
