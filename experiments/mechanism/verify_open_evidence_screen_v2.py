"""Independent offline readiness/completion verification of new retrieval screen."""
import io
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch
from experiments.mechanism import run_open_evidence_screen_v2 as r
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def main():
    before = frozen_hashes(); r.verify()
    corpus_meta = r.base.read(r.data.ENV / 'index_manifest.json')
    assert corpus_meta['documents'] == 5233329
    cases = r.base.read(r.OUT / 'public_cases.json'); labels = r.base.read(r.OUT / 'private_labels.json')
    assert len(cases) == len({c['id'] for c in cases}) == len({c['question'] for c in cases}) == 8
    source = r.data.parquet_module().read_table(r.data.RAW / 'dev_fullwiki.parquet', columns=['id', 'question', 'answer', 'context']).to_pylist()
    index = {v['id']: v for v in source}
    for c in cases:
        assert set(c) == {'id', 'question', 'native_documents'}
        original = index[c['id']]
        assert c['question'] == original['question'] and labels[c['id']] == original['answer']
        assert [d['title'] for d in c['native_documents']] == original['context']['title']
        assert [d['text'] for d in c['native_documents']] == [''.join(s) for s in original['context']['sentences']]
    names = ('test_open_evidence_screen_v2', 'test_evidence_acquisition', 'test_counterfactual_credit_audit',
        'test_sampling_coordination_screen', 'test_bandit_policy_learning', 'test_action_contribution_frontier',
        'test_option_binding_check', 'test_full_context_challenge', 'test_targeted_challenge_preflight',
        'test_targeted_challenge', 'test_document_verification', 'test_equilibrium_ranking_reference',
        'test_equilibrium_reference_pipeline', 'test_learned_production_game', 'test_learned_production_runner')
    final = (r.OUT / 'result.json').exists()
    with patch('socket.socket.connect', side_effect=AssertionError('offline')), patch('socket.getaddrinfo', side_effect=AssertionError('offline')):
        stream = io.StringIO()
        tests = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.' + n for n in names]))
        assert tests.wasSuccessful(), stream.getvalue()
        if final:
            outputs = r.base.read(r.OUT / 'sealed_outputs.json')
            wanted = {}
            def blocked(req): raise AssertionError('no new API calls during replay')
            for c in cases:
                assert r.collect(c, blocked, {'used': 0, 'cap': 5}) == outputs[c['id']]
                retrieval = r.base.read(r.OUT / c['id'] / 'retrieval.json')
                for env in retrieval['environments'].values():
                    assert len(env['searches']) <= (3 if env is retrieval['environments'].get('all') else 2)
                    for d in env['documents']: wanted[d['docid']] = d
                folder = r.OUT / c['id']; raws = r.read_rows(folder / 'responses.jsonl')
                assert len(raws) == len(r.read_rows(folder / 'attempts.jsonl')) <= 13
                by_key = {v['key']: v for v in raws}
                for stage in (*r.ANSWER_ARMS, 'plan', 'seq_plan'):
                    if stage not in by_key:
                        alias = r.base.read(folder / (stage + '_alias.json'))
                        assert alias['source_key'] in by_key
                        assert by_key[alias['source_key']]['request_hash'] == alias['request_hash']
                for row in raws:
                    data = json.loads(row['request']['messages'][1]['content'])
                    assert set(data) <= {'question', 'documents'}
                    assert data['question'] == c['question']
            # Verify all fetched source text against pinned corpus Parquet in one
            # streaming pass; do not merely trust a stored quote/hash field.
            found = set()
            for batch in r.data.parquet_module().ParquetFile(r.data.RAW / 'corpus.parquet').iter_batches(batch_size=10000):
                for d in batch.to_pylist():
                    key = str(d['_id'])
                    if key in wanted:
                        assert wanted[key]['text'] == d['text'] and wanted[key]['title'] == d['title']
                        found.add(key)
            assert found == set(wanted)
            with patch('builtins.print'): r.analyze()
    assert before == frozen_hashes()
    paths = [p for p in r.OUT.rglob('*') if p.is_file() and p.name not in ('run.lock', 'preparation_acceptance.json', 'final_acceptance.json')]
    paths += [Path(__file__)]
    accepted = {'status': 'open_retrieval_screen_completed' if final else 'open_retrieval_screen_prepared',
        'tests': tests.testsRun, 'verification_new_calls': 0, 'legacy_frozen_files': len(before),
        'corpus_documents': 5233329, 'n': 8,
        'sha256': {str(p.relative_to(r.ROOT)): r.base.h(p) for p in paths}}
    r.freeze(r.OUT / ('final_acceptance.json' if final else 'preparation_acceptance.json'), accepted)
    print(json.dumps({k: v for k, v in accepted.items() if k != 'sha256'}, ensure_ascii=False))


if __name__ == '__main__': main()
