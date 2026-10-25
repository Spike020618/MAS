"""Offline acceptance of the paid-run implementation, never starts paid work."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

from experiments.mechanism import run_full_context_challenge as run
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes


def main():
    before = frozen_hashes()
    protocol = run.verify(run.OUT)
    cases = run.read(run.OUT / 'public_cases.json')
    labels = run.read(run.OUT / 'private_labels.json')
    selection = run.read(run.OUT / 'selection.json')
    assert len(cases) == len({c['article_id'] for c in cases}) == len({c['article_sha256'] for c in cases}) == 8
    assert not {c['article_id'] for c in cases} & set(selection['excluded_article_ids'])
    assert not {c['article_sha256'] for c in cases} & set(selection['excluded_article_hashes'])
    for rel, digest in selection['journal_sha256'].items():
        assert run.h(run.ROOT / rel) == digest, rel
    with zipfile.ZipFile(run.ARCHIVE) as z:
        rows = [json.loads(line) for line in z.read('QuALITY.v1.0.1.htmlstripped.dev').decode().splitlines() if line.strip()]
    for c in cases:
        assert set(c) == {'id', 'article_id', 'article', 'article_sha256', 'question', 'choices'}
        matching = [(r, q) for r in rows if str(r['article_id']) == c['article_id']
                    for q in r['questions'] if q['question'] == c['question']]
        assert any(r['article'] == c['article'] and dict(zip('ABCD', q['options'])) == c['choices']
                   and 'ABCD'[int(q['gold_label']) - 1] == labels[c['id']] for r, q in matching)
        data = json.loads(run.req(c, None, 'native', {}, [])['messages'][1]['content'])
        assert set(data) == {'article', 'question', 'choices'}
    assert protocol['calls'] == 360
    assert not list(run.OUT.rglob('attempts.jsonl'))
    assert not list(run.OUT.rglob('responses.jsonl'))
    modules = ('test_full_context_challenge', 'test_targeted_challenge_preflight', 'test_targeted_challenge',
               'test_document_verification', 'test_equilibrium_ranking_reference',
               'test_equilibrium_reference_pipeline', 'test_learned_production_game', 'test_learned_production_runner')
    with patch('socket.socket.connect', side_effect=AssertionError('offline acceptance')), \
         patch('socket.getaddrinfo', side_effect=AssertionError('offline acceptance')):
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.' + name for name in modules]))
        assert result.wasSuccessful(), stream.getvalue()
    assert frozen_hashes() == before
    paths = [p for p in run.OUT.rglob('*') if p.is_file() and p.name not in ('run.lock', 'readiness_acceptance.json')]
    paths += [Path(__file__), run.ROOT / 'experiments/tests/test_full_context_challenge.py',
              run.ROOT / 'docs/research/全文竞争核验_启动准备验收_2026-10-25.md']
    run.freeze(run.OUT / 'readiness_acceptance.json', {
        'status': 'OFFLINE_ENGINEERING_READY_PAID_NOT_AUTHORIZED', 'new_api_calls': 0,
        'tests': result.testsRun, 'historical_frozen_files_preserved': len(before),
        'previous_journals_checked': len(selection['journal_sha256']),
        'source_groups': len(cases), 'planned_physical_call_limit': 360,
        'approval_file_created': False, 'scientific_efficacy_proven': False,
        'sha256': {str(p.relative_to(run.ROOT)): run.h(p) for p in paths}})
    print(json.dumps({'status': 'OFFLINE_ENGINEERING_READY_PAID_NOT_AUTHORIZED',
                      'tests': result.testsRun, 'new_api_calls': 0,
                      'historical_files_preserved': len(before)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
