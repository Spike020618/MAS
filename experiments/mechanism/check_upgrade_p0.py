"""Reproducible offline P0 gate. Synthetic tables are NOT quality experiments."""
import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path
import random
import time

from experiments.core.mechanism.upgrade_game import Contract, improve
from experiments.mechanism.audit_upgrade_game import audit


def run():
    rng = random.Random(20261021)
    runs = updates = 0
    worst_regret = 0.0
    started = time.perf_counter()
    for _ in range(100):
        contract = Contract(tuple(rng.random() for _ in range(8)),
                            tuple((0, rng.random() / 10) for _ in range(3)), 1, 'synthetic-property-check')
        for initial in contract.profiles():
            for order in permutations(range(3)):
                result = improve(contract, initial, order)
                checked = audit(contract, result['actions'])
                worst_regret = max(worst_regret, checked['selected']['regret'])
                assert checked['selected']['regret'] <= 1e-12
                assert len(result['trace']) <= 7
                for profile in contract.profiles():
                    before = contract.settlement(profile)
                    assert sum(before['payments']) <= contract.budget + 1e-12
                    for i in range(3):
                        changed = list(profile); changed[i] = 1 - changed[i]
                        after = contract.settlement(changed)
                        error = (after['utilities'][i] - before['utilities'][i]) - (after['potential'] - before['potential'])
                        assert abs(error) < 1e-12
                runs += 1
                updates += len(result['trace'])
    root = Path(__file__).resolve().parents[2]
    files = ['experiments/core/mechanism/upgrade_game.py',
             'experiments/core/mechanism/upgrade_execution.py',
             'experiments/mechanism/audit_upgrade_game.py',
             'experiments/mechanism/check_upgrade_p0.py']
    return {'scope': 'synthetic offline algebra/execution checks, not empirical efficacy',
            'seed': 20261021, 'tables': 100, 'runs': runs, 'accepted_updates': updates,
            'max_terminal_regret': worst_regret, 'wall_seconds': time.perf_counter() - started,
            'api_calls': 0, 'algebra_gate': 'passed',
            'pilot_gate': 'not_passed_data_and_transport_audit_pending',
            'files': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
