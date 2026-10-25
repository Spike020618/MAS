"""Offline acceptance of the bounded pilot; no effectiveness certification."""
from hashlib import sha256
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mechanism.run_premise_review import OUT, p, parse_row, initial_request
from experiments.mechanism import premise_review as method
from experiments.mechanism.run_upgrade_pilot import CallJournal, read_rows, freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.complete_budgeted_information import verify_prior_snapshots


def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    for rel, h in protocol['code_sha256'].items():
        assert sha256((p.ROOT/rel).read_bytes()).hexdigest() == h
    prov = json.loads((OUT/'provenance.json').read_text())
    for name, h in prov['files'].items():
        assert sha256((OUT/name).read_bytes()).hexdigest() == h
    cases = json.loads((OUT/'public_cases.json').read_text())
    assert p.digest(cases) == protocol['cases_hash'] and len(cases) == 24
    initial = json.loads((OUT/'sealed_initial.json').read_text())
    reviews = json.loads((OUT/'sealed_reviews.json').read_text())
    rows = read_rows(OUT/'responses.jsonl')
    assert len(rows) <= 240
    def forbidden(_):
        raise AssertionError('acceptance must not send requests')
    journal = CallJournal(OUT, forbidden, 240)
    with patch('socket.socket.connect', side_effect=AssertionError('offline')), \
         patch('socket.getaddrinfo', side_effect=AssertionError('offline')):
        log = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            'experiments.tests.test_premise_runner', 'experiments.tests.test_premise_review',
            'experiments.tests.test_interface_verdict_audit', 'experiments.tests.test_legacy_gain_attribution'])
        tested = unittest.TextTestRunner(stream=log).run(suite)
        assert tested.wasSuccessful(), log.getvalue()
        for case in cases:
            for agent in range(3):
                r = journal.call(case['id']+'/initial/'+str(agent), initial_request(case, agent, protocol['model']))
                assert parse_row(r, case['choices']) == initial[case['id']][agent]
            reports = [{'answer': r['answer'], 'reasoning': r['reasoning']} for r in initial[case['id']]]
            for arm in method.ARMS:
                req = {'backend': 'deepseek', **method.request(case['question'], case['choices'], reports, arm, protocol['model'])}
                r = journal.call(case['id']+'/review/'+arm, req)
                assert parse_row(r, case['choices']) == reviews[case['id']][arm]
        recomputed = method.paired_summary(json.loads((OUT/'scoring_records.json').read_text()))
        result = json.loads((OUT/'result.json').read_text())
        for k,v in recomputed.items():
            assert result[k] == v
    assert len(rows) == 120, 'retry costs need additional audit before accepting this exact no-retry run'
    assert all(r['response']['http_status'] == 200 for r in rows)
    for row in rows:
        b = json.loads(row['response']['body'])
        u = b['usage']
        assert u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens']
        assert b['model'] and row['wall_seconds'] >= 0
    attempts = read_rows(OUT/'attempts.jsonl')
    assert len(attempts) == 120
    first_review = next(i for i,r in enumerate(attempts) if '/review/' in r['key'])
    assert first_review == 72 and all('/initial/' in r['key'] for r in attempts[:72])
    original = frozen_hashes()
    prior = verify_prior_snapshots()
    files = [f for f in OUT.iterdir() if f.is_file() and f.name not in ('run.lock', 'acceptance.json')]
    files += [Path(__file__), p.ROOT/'experiments/mechanism/report_premise_pilot.py',
              p.ROOT/'experiments/tests/test_premise_runner.py', p.ROOT/'experiments/tests/test_premise_review.py',
              p.ROOT/'docs/research/关键前提核查_小批量结果_2026-10-24.md']
    freeze(OUT/'acceptance.json', {'status': 'execution_verified_not_efficacy_or_equilibrium_proven',
           'new_model_calls_this_pilot': 120, 'new_model_calls_during_acceptance': 0,
           'tests_passed': tested.testsRun, 'test_log': log.getvalue(), 'frozen_original_files_verified': len(original),
           'prior_snapshots_verified': len(prior), 'sealed_initial_before_review': True,
           'sha256': {str(f.relative_to(p.ROOT)): sha256(f.read_bytes()).hexdigest() for f in files}})
    print('Acceptance: 120 raw calls replayed offline; costs consistent; 38 tests passed; old evidence preserved.')


if __name__ == '__main__':
    main()
