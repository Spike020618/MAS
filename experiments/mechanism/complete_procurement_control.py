"""Final necessity-control acceptance, preserving the previous mutable entry."""
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


def fingerprint(path): return sha256(path.read_bytes()).hexdigest()


def main():
    output = io.StringIO()
    with patch('socket.getaddrinfo', side_effect=RuntimeError('verification forbids DNS')), \
         patch('socket.socket.connect', side_effect=RuntimeError('verification forbids network')), \
         patch('socket.socket.connect_ex', side_effect=RuntimeError('verification forbids network')):
        with redirect_stdout(output), redirect_stderr(output):
            base = complete_base()
            for name in ('audit_effort_generalization', 'audit_certified_procurement_control'):
                runpy.run_module('experiments.mechanism.'+name, run_name='__main__')
            suite = unittest.defaultTestLoader.loadTestsFromNames([
                'experiments.tests.test_effort_generalization',
                'experiments.tests.test_certified_procurement_control'])
            tests = unittest.TextTestRunner(stream=output).run(suite)
        if not tests.wasSuccessful(): raise ValueError(output.getvalue())
    # Earlier acceptance includes the then-current mutable work entry. Preserve
    # and verify its exact bytes through its saved text, not by pretending the
    # new entry is unchanged. Every scientific source remains byte-identical.
    previous = json.loads((AUDIT/'acceptance.json').read_text())
    checkpoint = json.loads((AUDIT/'active_plan_before_procurement_control.json').read_text())
    for rel, expected in previous['snapshot_sha256'].items():
        if rel == checkpoint['path']:
            if sha256(checkpoint['utf8_text'].encode()).hexdigest() != expected or checkpoint['sha256'] != expected:
                raise ValueError('previous mutable entry snapshot not preserved')
        elif fingerprint(ROOT/rel) != expected:
            raise ValueError('previous scientific scope snapshot changed: '+rel)
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    if diff.returncode: raise ValueError(diff.stdout+diff.stderr)
    paths = [AUDIT/'acceptance.json', OUT/'acceptance.json', AUDIT/'procurement_control.json',
        AUDIT/'active_plan_before_procurement_control.json',
        ROOT/'experiments/mechanism/audit_certified_procurement_control.py',
        ROOT/'experiments/mechanism/complete_procurement_control.py',
        ROOT/'experiments/tests/test_certified_procurement_control.py',
        ROOT/'docs/research/独立预算研究_最终必要性对照_2026-10-23.md',
        ROOT/'docs/research/ACTIVE_RESEARCH_PLAN.md']
    for path in paths:
        if path.suffix in ('.py', '.md'):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if line != line.rstrip(): raise ValueError(f'trailing whitespace: {path}:{i}')
    acceptance = {'status': 'OFFLINE_RESEARCH_AUDIT_COMPLETE_COMPLEX_PAYMENT_NECESSITY_NOT_ESTABLISHED',
        'new_model_calls': 0, 'network_and_dns_forbidden': True,
        'tests_passed': base['tests_passed']+tests.testsRun,
        'old_frozen_files_verified': base['original_frozen_files_verified'],
        'prior_study_snapshots_verified': len(base['prior_acceptance_snapshots_sha256']),
        'previous_work_entry_preserved_with_exact_hash': True,
        'previous_scientific_outputs_unchanged': True, 'git_diff_check_passed': True,
        'restricted_effort_game_structure_valid': True,
        'simple_contract_reproduces_same_actions': True,
        'complex_payment_advantage_or_population_equilibrium': False,
        'no_new_paid_collection_or_document_revision': True,
        'snapshot_sha256': {str(p.relative_to(ROOT)): fingerprint(p) for p in paths}}
    freeze(AUDIT/'procurement_control_acceptance.json', acceptance)
    print(json.dumps({k: v for k, v in acceptance.items() if k != 'snapshot_sha256'}, ensure_ascii=False, indent=2))
    return acceptance


if __name__ == '__main__': main()
