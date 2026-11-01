"""Independent offline acceptance; no transport and no source modifications."""
import io
import json
import socket
import subprocess
import unittest
from unittest.mock import patch
from experiments.mechanism import run_action_comparison as r
from experiments.mechanism import redesign_hotpot_data as data
from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def main():
    legacy = frozen_hashes()
    protocol = r.verify()
    cases = r.read(r.OUT / 'public_cases.json')
    outputs = r.read(r.OUT / 'sealed_outputs.json')
    if len(outputs) != 24 or set(outputs) != {c['id'] for c in cases}:
        raise ValueError('complete collection required')
    before = {str(p.relative_to(r.ROOT)): r.sha(p) for p in r.OUT.rglob('*')
              if p.is_file() and p.suffix in ('.json', '.jsonl') and p.name != 'acceptance.json'}
    stream = io.StringIO()
    with (patch.object(socket.socket, 'connect', side_effect=AssertionError('network prohibited')),
          patch.object(socket, 'getaddrinfo', side_effect=AssertionError('network prohibited'))):
        tests = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames([
            'experiments.tests.test_action_comparison', 'experiments.tests.test_action_targets',
            'experiments.tests.test_sequential_allocation_screen', 'experiments.tests.test_sequential_cost_correction']))
        if not tests.wasSuccessful():
            raise AssertionError(stream.getvalue())
        rows = r.ledger_rows(r.OUT)
        if len(rows) > protocol['maximum_physical_calls']:
            raise ValueError('request limit violated')
        prices = r.read(r.OUT / 'prices.json')
        commitment = r.read(r.OUT / 'price_commitment.json')
        receipt = r.read(r.OUT / 'collection_receipt.json')
        actual_proxy = sum(r.cost(v, prices) for v in rows)
        if commitment['sha256'] != r.sha(r.OUT / 'prices.json') or commitment['prices'] != prices:
            raise ValueError('price commitment changed')
        if actual_proxy > protocol['usd_proxy_cap'] or abs(actual_proxy - receipt['usd_proxy']) > 1e-12:
            raise ValueError('cost receipt/cap mismatch')
        wanted = {}
        for case in cases:
            cid = case['id']
            output = outputs[cid]
            if r.read(r.OUT / cid / 'sealed_outputs.json') != output:
                raise ValueError('case seal mismatch')
            r.replay_case(case, output, r.read_rows(r.OUT / cid / 'responses.jsonl'))
            env = output['arms']['retrieve']['retrieval']
            if env:
                for d in env['documents']:
                    wanted[d['docid']] = d
        corpus = r.Corpus(r.INDEX)
        try:
            for key, doc in wanted.items():
                found = corpus.con.execute('SELECT title,text FROM docs WHERE docid=?', (key,)).fetchone()
                if found is None or tuple(found) != (doc['title'], doc['text']):
                    raise ValueError('retrieved source absent/changed in corpus')
        finally:
            corpus.close()
        selection = r.read(r.OUT / 'selection.json')
        if r.sha(data.RAW / 'dev_fullwiki.parquet') != selection['raw_dataset_sha256']:
            raise ValueError('original dataset changed')
        original = {v['id']: v for v in data.parquet_module().read_table(data.RAW / 'dev_fullwiki.parquet').to_pylist()}
        labels, splits = r.read(r.OUT / 'private_labels.json'), r.read(r.OUT / 'splits.json')
        pages = {s: set() for s in ('development', 'confirmation')}
        for case in cases:
            src = original[case['id']]
            if case['question'] != src['question'] or labels[case['id']]['answer'] != src['answer']:
                raise ValueError('source question/answer mismatch')
            if labels[case['id']]['supporting_facts'] != src['supporting_facts']:
                raise ValueError('support annotations changed')
            if [(d['title'], d['text']) for d in case['native_documents']] != list(zip(src['context']['title'], [''.join(v) for v in src['context']['sentences']])):
                raise ValueError('native sources changed')
            pages[splits[case['id']]].update(src['supporting_facts']['title'])
        if pages['development'] & pages['confirmation']:
            raise ValueError('support-page split overlap')
        from experiments.mechanism import score_action_comparison as scorer
        with patch('builtins.print'):
            scorer.main()
    if legacy != frozen_hashes():
        raise ValueError('legacy frozen evidence changed')
    previous = r.read(r.ROOT / 'experiments/results/redesign_v1/hour_acceptance_v1/acceptance.json')['sha256']
    if any(r.sha(r.ROOT / p) != value for p, value in previous.items()):
        raise ValueError('previous acceptance evidence changed')
    if any(r.sha(r.ROOT / p) != value for p, value in before.items()):
        raise ValueError('offline acceptance modified collection/scoring artifacts')
    subprocess.run(['git', 'diff', '--check'], cwd=r.ROOT, check=True)
    result = {'status': 'offline_replay_verified_NOT_proof_of_effect', 'tests_passed': tests.testsRun,
        'n': len(cases), 'new_verification_api_calls': 0, 'physical_calls': len(rows),
        'physical_tokens': sum(r.usage(v)['total_tokens'] for v in rows), 'usd_proxy': actual_proxy,
        'retrieved_documents_source_checked': len(wanted), 'legacy_files_unchanged': len(legacy),
        'previous_acceptance_files_unchanged': len(previous), 'sha256': before,
        'verifier_sha256': r.sha(__import__('pathlib').Path(__file__))}
    freeze(r.OUT / 'acceptance.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
