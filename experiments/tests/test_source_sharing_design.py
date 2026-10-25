"""Executable design checks, not a production settlement or empirical study."""
from itertools import product
import random
import unittest


def evaluate(profile, bundles, weights, costs):
    selected = [set(bundles[i]) if action else set()
                for i, action in enumerate(profile)]
    counts = {s: sum(s in bundle for bundle in selected) for s in weights}
    payments = [sum(weights[s] / counts[s] for s in bundle) for bundle in selected]
    utilities = [payments[i] - costs[i] * profile[i] for i in range(len(profile))]
    potential = sum(weights[s] * sum(1 / k for k in range(1, n + 1))
                    for s, n in counts.items()) - sum(
                        costs[i] * profile[i] for i in range(len(profile)))
    return payments, utilities, potential


class SourceSharingDesignTests(unittest.TestCase):
    def test_potential_identity_budget_and_existence(self):
        rng = random.Random(191019)
        for _ in range(100):
            weights = {s: rng.random() for s in range(4)}
            bundles = [[s for s in weights if rng.random() < .5] for _ in range(3)]
            costs = [rng.random() for _ in range(3)]
            table = {p: evaluate(p, bundles, weights, costs)
                     for p in product((0, 1), repeat=3)}
            equilibria = []
            for p, (payments, utilities, potential) in table.items():
                self.assertLessEqual(sum(payments), sum(weights.values()) + 1e-12)
                gains = []
                for i in range(3):
                    other = list(p)
                    other[i] = 1 - p[i]
                    _, alternative, phi = table[tuple(other)]
                    gain = alternative[i] - utilities[i]
                    self.assertAlmostEqual(phi - potential, gain)
                    gains.append(gain)
                if max(gains) <= 1e-12:
                    equilibria.append(p)
            self.assertTrue(equilibria)
            # Fair round-robin strict improvements terminate; ties stay put.
            for initial in table:
                current = initial
                seen = {current}
                while True:
                    changed = False
                    for i in range(3):
                        other = list(current)
                        other[i] = 1 - current[i]
                        other = tuple(other)
                        if table[other][1][i] > table[current][1][i] + 1e-12:
                            self.assertNotIn(other, seen)
                            seen.add(other)
                            current = other
                            changed = True
                    if not changed:
                        break
                self.assertIn(current, equilibria)

    def test_unclaimed_budget_not_redistributed_and_duplicates_ignored(self):
        payments, _, _ = evaluate((1, 0), [[0, 0], [1]], {0: 2, 1: 3}, [0, 0])
        self.assertEqual(payments, [2, 0])

    def test_synchronous_best_response_can_cycle(self):
        # w/2 < cost < w: both enter when alone, both leave when together.
        def simultaneous(profile):
            result = []
            for i in range(2):
                candidates = []
                for action in (0, 1):
                    trial = list(profile)
                    trial[i] = action
                    utility = evaluate(trial, [[0], [0]], {0: 1}, [.75, .75])[1][i]
                    candidates.append(utility)
                result.append(int(candidates[1] > candidates[0]))
            return tuple(result)
        self.assertEqual(simultaneous((0, 0)), (1, 1))
        self.assertEqual(simultaneous((1, 1)), (0, 0))


if __name__ == '__main__':
    unittest.main()
