"""No-network reproducibility and bounded acceptance for this research hour.

Does not certify effectiveness, population equilibrium or online API execution.
Never changes old source traces, protected modules, papers or system files.
"""
from contextlib import redirect_stdout, redirect_stderr
from hashlib import sha256
import io
import json
from pathlib import Path
import runpy
import subprocess
import unittest
from unittest.mock import patch

from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.replay_budgeted_information import ROOT, OUT
from experiments.mechanism.run_upgrade_pilot import freeze

MODULES = ['replay_budgeted_information', 'diagnose_budgeted_information',
           'verify_bayesian_information_fixture', 'replay_certified_effort_game',
           'diagnose_certified_effort_game']
TESTS = ['test_budgeted_information_sim', 'test_budgeted_information_replay',
    'test_bayesian_information_fixture', 'test_certified_effort_game',
    'test_incentive_necessity', 'test_incentive_resource_profiles', 'test_routing_holdout',
    'test_probability_compatibility', 'test_probability_routing', 'test_dispute_review',
    'test_aeic_context_probe', 'test_low_cost_router', 'test_resource_allocation_probe',
    'test_upgrade_pilot', 'test_upgrade_v3']
NEW_CODE = ['experiments/core/mechanism/budgeted_information_sim.py',
    'experiments/core/mechanism/bayesian_information_fixture.py',
    'experiments/core/mechanism/certified_effort_game.py',
    *['experiments/mechanism/'+name+'.py' for name in MODULES],
    'experiments/mechanism/complete_budgeted_information.py',
    *['experiments/tests/'+name+'.py' for name in TESTS[:4]]]
REPORT = 'docs/research/独立预算与可验证投入_研究验收_2026-10-23.md'


def fingerprint(path): return sha256(path.read_bytes()).hexdigest()


def verify_prior_snapshots():
    paths = [ROOT/'experiments/results/resource_allocation'/name/'acceptance.json'
        for name in ['probability_routing_v1', 'routing_holdout_validation_v1', 'aeic_context_v1', 'dispute_review_v1']]
    paths.append(ROOT/'experiments/results/audit/incentive_necessity_2026-10-23/acceptance.json')
    verified = {}
    for path in paths:
        value = json.loads(path.read_text())
        for rel, expected in value['snapshot_sha256'].items():
            if fingerprint(ROOT/rel) != expected: raise ValueError('prior snapshot changed: '+rel)
        verified[str(path.relative_to(ROOT))] = fingerprint(path)
    return verified


def audit_seals_and_ledgers():
    generic = json.loads((OUT/'episode_traces.json').read_text())
    count = 0
    for rows in generic.values():
        for row in rows:
            data = row['settlement']
            events = data['events']
            if [e['event'] for e in events] != ['sealed']*3+['aggregate', 'delayed_settlement']:
                raise ValueError('seal/aggregate/outcome order changed')
            from experiments.core.mechanism.budgeted_information_sim import digest
            if data['aggregate_commitment'] != digest(events[:-1]): raise ValueError('aggregate seal changed')
            for record in data['records']:
                fields = ('agent', 'action', 'report', 'tokens', 'initial_tokens', 'receipt',
                          'accepted_answer', 'rejected_inconsistent_conclusion')
                base = {k: record[k] for k in fields}
                if events[record['agent']]['commitment'] != digest(base): raise ValueError('report seal changed')
                if record['wallet_remaining'] != 3000-record['tokens']: raise ValueError('wallet mismatch')
                if abs(record['utility']-(record['credit']-record['tokens'])) > 1e-8: raise ValueError('utility mismatch')
                count += 1
    result = json.loads((OUT/'certified_effort_result.json').read_text())
    if result['new_api_calls'] != 0: raise ValueError('unexpected API usage')
    traces = json.loads((OUT/'certified_effort_episode_traces.json').read_text())
    for scale, rows in traces.items():
        for row in rows:
            if not 0 <= row['credits_total'] <= 2*int(scale): raise ValueError('pivotal payment budget')
            for r in row['ledgers']:
                if r['utility'] != r['credit']-r['tokens']: raise ValueError('certified utility mismatch')
                if r['wallet_remaining'] != 3000-r['tokens']: raise ValueError('certified wallet mismatch')
    return count


def main():
    original = frozen_hashes()
    prior = verify_prior_snapshots()
    output = io.StringIO()
    with patch('socket.getaddrinfo', side_effect=RuntimeError('offline verification forbids DNS')), \
         patch('socket.socket.connect', side_effect=RuntimeError('offline verification forbids network')), \
         patch('socket.socket.connect_ex', side_effect=RuntimeError('offline verification forbids network')):
        for name in MODULES:
            with redirect_stdout(output), redirect_stderr(output):
                runpy.run_module('experiments.mechanism.'+name, run_name='__main__')
        suite = unittest.defaultTestLoader.loadTestsFromNames(['experiments.tests.'+m for m in TESTS])
        tests = unittest.TextTestRunner(stream=output, verbosity=1).run(suite)
        if not tests.wasSuccessful(): raise ValueError(output.getvalue())
        sealed_records = audit_seals_and_ledgers()
    if frozen_hashes() != original or verify_prior_snapshots() != prior:
        raise ValueError('protected evidence changed')
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    if diff.returncode: raise ValueError(diff.stdout+diff.stderr)
    # git diff ignores untracked files; check every newly written source too.
    for rel in NEW_CODE+[REPORT]:
        for i, line in enumerate((ROOT/rel).read_text().splitlines(), 1):
            if line.rstrip() != line: raise ValueError(f'trailing whitespace: {rel}:{i}')
    sources = {ROOT/rel for rel in NEW_CODE+[REPORT]}
    sources.update(f for f in OUT.glob('*.json') if f.name != 'acceptance.json')
    acceptance = {'status': 'OFFLINE_ENVIRONMENT_AND_RESTRICTED_CALIBRATION_GAME_VERIFIED_NOT_RESEARCH_COMPLETION',
        'new_api_calls': 0, 'network_and_dns_forbidden': True,
        'tests_passed': tests.testsRun, 'individual_sealed_records_rechecked': sealed_records,
        'original_frozen_files_verified': len(original), 'prior_acceptance_snapshots_sha256': prior,
        'reproduced_modules': MODULES, 'git_diff_check_passed': True, 'new_file_whitespace_check_passed': True,
        'programmatic_actors_not_rational_llm_claim': True,
        'restricted_effort_game_has_exact_potential': True,
        'complete_free_report_or_participation_mechanism': False,
        'population_equilibrium_or_performance_superiority': False,
        'paid_online_execution_verified': False,
        'snapshot_sha256': {str(f.relative_to(ROOT)): fingerprint(f) for f in sorted(sources)}}
    freeze(OUT/'acceptance.json', acceptance)
    print(json.dumps({k: v for k, v in acceptance.items() if k != 'snapshot_sha256'}, ensure_ascii=False, indent=2))
    return acceptance


if __name__ == '__main__': main()
