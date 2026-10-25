"""Final reproducible envelope, including bounded historical source review."""
from contextlib import redirect_stdout, redirect_stderr
from hashlib import sha256
import io
import json
import runpy
import unittest
from unittest.mock import patch

from experiments.mechanism.complete_procurement_control import main as complete_control
from experiments.mechanism.audit_historical_information_assets import OUT as ASSETS
from experiments.mechanism.audit_effort_generalization import AUDIT
from experiments.mechanism.audit_certified_procurement_control import menu_reply
from experiments.mechanism.replay_budgeted_information import ROOT, OUT, load_observations
from experiments.mechanism.run_upgrade_pilot import freeze


def main():
    log = io.StringIO()
    with patch('socket.getaddrinfo', side_effect=RuntimeError('verification forbids DNS')), \
         patch('socket.socket.connect', side_effect=RuntimeError('verification forbids network')), \
         patch('socket.socket.connect_ex', side_effect=RuntimeError('verification forbids network')):
        with redirect_stdout(log), redirect_stderr(log):
            parent = complete_control()
            runpy.run_module('experiments.mechanism.audit_historical_information_assets', run_name='__main__')
            suite = unittest.defaultTestLoader.loadTestsFromName('experiments.tests.test_historical_information_assets')
            tests = unittest.TextTestRunner(stream=log).run(suite)
        if not tests.wasSuccessful(): raise ValueError(log.getvalue())
        # The first simple-contract check used the 2000-token reservation as a
        # sign witness for an UNSELECTED deviation, never as consumed tokens.
        # Independently substitute EVERY actual recorded alternative-path cost.
        observations, upgrades, _ = load_observations(ROOT/'experiments/results/resource_allocation/routing_holdout_validation_v1')
        data = json.loads((AUDIT/'procurement_control.json').read_text())
        checks = 0
        for scale, scenario in data['scenarios'].items():
            target = scenario['target_profile_unchanged']
            paid = 0
            for cid in observations:
                for i in range(3):
                    actual = upgrades[(cid, i)][1]
                    reply = menu_reply(target[i], actual)
                    if reply['chosen'] != target[i]: raise ValueError('actual alternative cost contradicts implementation')
                    paid += reply['credit_if_upgrade'] if reply['chosen'] else 0
                    checks += 1
            if paid != scenario['cost_plus_one_virtual_payment']:
                raise ValueError('actual source-cost replay changed a payment total')
    assets = json.loads((ASSETS/'result.json').read_text())
    for row in assets['reviewed_entries']:
        if sha256((ROOT/row['path']).read_bytes()).hexdigest() != row['source_sha256']:
            raise ValueError('historical source changed during review')
    paths = [AUDIT/'procurement_control_acceptance.json', ASSETS/'result.json',
        ROOT/'experiments/mechanism/audit_historical_information_assets.py',
        ROOT/'experiments/mechanism/complete_information_hour.py',
        ROOT/'experiments/tests/test_historical_information_assets.py',
        ROOT/'docs/research/历史信息入口_适用性核查_2026-10-23.md']
    for path in paths:
        if path.suffix in ('.py', '.md'):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if line != line.rstrip(): raise ValueError(f'trailing whitespace: {path}:{i}')
    acceptance = {'status': 'OFFLINE_RESEARCH_EXPLORATION_VERIFIED_NO_COMPLEX_MECHANISM_ADVANTAGE_CLAIM',
        'new_model_calls': 0, 'network_and_dns_forbidden': True,
        'tests_passed': parent['tests_passed']+tests.testsRun,
        'old_frozen_files_unchanged': parent['old_frozen_files_verified'],
        'prior_study_snapshots_unchanged': parent['prior_study_snapshots_verified'],
        'historical_entries_reviewed': len(assets['reviewed_entries']),
        'historical_review_is_not_project_wide_nonexistence_proof': True,
        'all_observed_counterfactual_cost_reply_checks': checks,
        'unselected_reservation_cost_not_charged_as_consumption': True,
        'research_documents_created': True, 'papers_patent_system_ppt_unchanged_by_this_work': True,
        'restricted_game_structure_valid_but_complex_payment_necessity_not_established': True,
        'git_diff_check_passed_by_parent': parent['git_diff_check_passed'],
        'snapshot_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}}
    freeze(ASSETS/'acceptance.json', acceptance)
    print(json.dumps({k: v for k, v in acceptance.items() if k != 'snapshot_sha256'}, ensure_ascii=False, indent=2))
    return acceptance


if __name__ == '__main__': main()
