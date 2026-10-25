"""Offline acceptance: software and legacy preservation, not efficacy."""
from hashlib import sha256
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mechanism.run_upgrade_pilot import freeze
from experiments.mechanism.complete_legacy_gain_audit import frozen_hashes
from experiments.mechanism.complete_budgeted_information import verify_prior_snapshots
from experiments.mechanism.replay_equilibrium_reference import main as replay, OUT, ROOT


def main():
    original = frozen_hashes(); prior = verify_prior_snapshots()
    # Preserve newest crossed development evidence, not just oldest baseline.
    previous = ROOT/'experiments/results/resource_allocation/verification_crossed_dev_v1/acceptance.json'
    acceptance = json.loads(previous.read_text())
    for rel, expected in acceptance['sha256'].items():
        assert sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
    names = ('test_equilibrium_ranking_reference','test_equilibrium_reference_pipeline',
             'test_signaling_game','test_token_signaling_adapter','test_scored_signaling',
             'test_verification_risk_game','test_verification_verdict_sensitivity')
    with (patch('socket.socket.connect', side_effect=AssertionError('offline only')),
          patch('socket.getaddrinfo', side_effect=AssertionError('offline only'))):
        with patch('builtins.print'): replay()
        log = io.StringIO()
        tested = unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['experiments.tests.'+name for name in names]))
        assert tested.wasSuccessful(), log.getvalue()
    assert frozen_hashes() == original
    files = [ROOT/'experiments/core/mechanism/equilibrium_ranking_reference.py',
             ROOT/'experiments/core/mechanism/sequence_likelihood.py',
             ROOT/'experiments/mechanism/run_equilibrium_reference.py',
             ROOT/'experiments/mechanism/replay_equilibrium_reference.py', Path(__file__),
             ROOT/'experiments/tests/test_equilibrium_ranking_reference.py',
             ROOT/'experiments/tests/test_equilibrium_reference_pipeline.py',
             ROOT/'docs/research/Consensus_Game复现入口_2026-10-25.md',
             ROOT/'docs/research/Consensus_Game实现验收_2026-10-25.md',
             OUT/'result.json', OUT/'sealed_readouts.json']
    result = {'status':'software_verified_full_text_empirical_reproduction_pending',
              'new_api_calls':0, 'tests':tested.testsRun, 'original_frozen_files':len(original),
              'prior_snapshots':len(prior), 'crossed_acceptance_sha256':sha256(previous.read_bytes()).hexdigest(),
              'sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},
              'limits':['no model runtime installed; HF scorer only covered by non-model pipeline tests',
                        'old token replay is post-hoc and not full sequence or independent efficacy']}
    freeze(OUT/'acceptance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'}, indent=2))


if __name__ == '__main__': main()
