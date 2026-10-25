"""Final envelope: reproduce accepted base and verify the scope addendum offline."""
from contextlib import redirect_stdout, redirect_stderr
from hashlib import sha256
import io
import json
import runpy
import subprocess
import unittest
from unittest.mock import patch

from experiments.mechanism.complete_budgeted_information import main as complete_base
from experiments.mechanism.audit_effort_generalization import AUDIT
from experiments.mechanism.replay_budgeted_information import ROOT, OUT
from experiments.mechanism.run_upgrade_pilot import freeze


def main():
    log = io.StringIO()
    with patch('socket.getaddrinfo', side_effect=RuntimeError('verification forbids DNS')), \
         patch('socket.socket.connect', side_effect=RuntimeError('verification forbids network')), \
         patch('socket.socket.connect_ex', side_effect=RuntimeError('verification forbids network')):
        with redirect_stdout(log), redirect_stderr(log):
            base = complete_base()
            runpy.run_module('experiments.mechanism.audit_effort_generalization', run_name='__main__')
            suite = unittest.defaultTestLoader.loadTestsFromName('experiments.tests.test_effort_generalization')
            tests = unittest.TextTestRunner(stream=log).run(suite)
        if not tests.wasSuccessful(): raise ValueError(log.getvalue())
    for rel, expected in base['snapshot_sha256'].items():
        if sha256((ROOT/rel).read_bytes()).hexdigest() != expected:
            raise ValueError('accepted base changed: '+rel)
    paths = [OUT/'acceptance.json', AUDIT/'result.json',
        ROOT/'experiments/mechanism/audit_effort_generalization.py',
        ROOT/'experiments/mechanism/complete_effort_hour.py',
        ROOT/'experiments/tests/test_effort_generalization.py',
        ROOT/'docs/research/投入均衡泛化_统计与因果边界_2026-10-23.md',
        ROOT/'docs/research/ACTIVE_RESEARCH_PLAN.md']
    for p in paths:
        if p.suffix in ('.py', '.md'):
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if line != line.rstrip(): raise ValueError(f'trailing whitespace: {p}:{i}')
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    if diff.returncode: raise ValueError(diff.stdout+diff.stderr)
    acceptance = {'status': 'RESEARCH_HOUR_OFFLINE_DELIVERABLES_VERIFIED_NOT_FULL_METHOD_ACCEPTANCE',
        'new_model_calls': 0, 'network_and_dns_forbidden': True,
        'base_tests': base['tests_passed'], 'addendum_tests': tests.testsRun,
        'total_tests_passed': base['tests_passed']+tests.testsRun,
        'old_frozen_files_unchanged': base['original_frozen_files_verified'],
        'prior_study_snapshots_unchanged': len(base['prior_acceptance_snapshots_sha256']),
        'base_sealed_records_rechecked': base['individual_sealed_records_rechecked'],
        'git_diff_check_passed': True,
        'strict_result_scope': 'authenticated-output K/U action game; exact calibration NE, not true population or free-report equilibrium',
        'known_open_gaps': ['stable superiority over simple incentives and direct strong baseline',
            'out-of-sample expected-utility/NE generalization', 'voluntary entry and private-cost reporting',
            'no causal-deletion proof for peer-guided multi-round updates', 'online provider execution not run'],
        'snapshot_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}}
    freeze(AUDIT/'acceptance.json', acceptance)
    print(json.dumps({k: v for k, v in acceptance.items() if k != 'snapshot_sha256'}, ensure_ascii=False, indent=2))
    return acceptance


if __name__ == '__main__': main()
