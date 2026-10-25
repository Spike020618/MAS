"""Offline acceptance of redesigned objective, not real-model superiority."""
from hashlib import sha256
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.complete_budgeted_information import verify_prior_snapshots
from experiments.mechanism.run_upgrade_pilot import freeze

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/results/audit/verification_risk_design_v1'


def fingerprint(p):return sha256(p.read_bytes()).hexdigest()


def main():
    frozen=frozen_hashes();prior=verify_prior_snapshots();verified=[]
    for rel in ('experiments/results/audit/signaling_foundation_v1/acceptance.json',
                'experiments/results/resource_allocation/signaling_pilot_v1/acceptance.json',
                'experiments/results/audit/signaling_failure_repair_v2/acceptance.json'):
        record=json.loads((ROOT/rel).read_text())
        for p,h in record['sha256'].items():assert fingerprint(ROOT/p)==h,p
        verified.append({'path':rel,'sha256':fingerprint(ROOT/rel)})
    names=['test_two_backend_complementarity','test_information_complementarity','test_simple_readout_crossfit',
        'test_thinking_control','test_complementarity_extraction','test_premise_review','test_premise_runner',
        'test_interface_verdict_audit','test_legacy_gain_attribution','test_reliable_readout','test_reliable_readout_runner',
        'test_signaling_game','test_signaling_interface','test_token_signaling_adapter','test_scored_signaling',
        'test_signaling_safety_audit','test_verification_risk_game']
    with patch('socket.socket.connect',side_effect=AssertionError('offline')),patch('socket.getaddrinfo',side_effect=AssertionError('offline')):
        from experiments.mechanism.prototype_verification_risk import main as prototype
        with patch('builtins.print'):prototype()
        log=io.StringIO();tested=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.'+name for name in names]))
        assert tested.wasSuccessful(),log.getvalue()
    assert frozen_hashes()==frozen
    files=[ROOT/'experiments/core/mechanism/verification_risk_game.py',
        ROOT/'experiments/tests/test_verification_risk_game.py',
        ROOT/'experiments/mechanism/prototype_verification_risk.py',Path(__file__),
        ROOT/'docs/research/真实核验风险博弈_目标重设计与原型_2026-10-24.md']
    files += [p for p in OUT.glob('*.json') if p.name!='acceptance.json']
    acceptance={'status':'new_objective_and_exact_empirical_game_solver_only_not_efficacy_validated',
        'tests_passed':tested.testsRun,'test_log':log.getvalue(),'original_frozen_files_verified':len(frozen),
        'prior_snapshots_verified':len(prior),'recent_acceptances_verified':verified,
        'new_api_calls':0,'new_independent_questions':0,
        'sha256':{str(p.relative_to(ROOT)):fingerprint(p) for p in files}}
    freeze(OUT/'acceptance.json',acceptance)
    print(json.dumps({k:v for k,v in acceptance.items() if k not in ('sha256','test_log','recent_acceptances_verified')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
