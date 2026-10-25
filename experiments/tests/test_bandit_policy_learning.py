import unittest
from math import log
from experiments.core.mechanism.bandit_policy_learning import IndependentActors, empirical_deviation


class BanditPolicyTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        actor = IndependentActors([3]); actor.logits = [[.2, -.1, .4]]
        analytic = actor.score_gradient(actor.probabilities()[0], 1, .7)
        for j in range(3):
            actor.logits[0][j] += 1e-6; plus = .7 * log(actor.probabilities()[0][1])
            actor.logits[0][j] -= 2e-6; minus = .7 * log(actor.probabilities()[0][1])
            actor.logits[0][j] += 1e-6
            self.assertAlmostEqual(analytic[j], (plus - minus) / 2e-6, places=7)

    def test_learning_changes_probabilities_and_freeze_blocks_updates(self):
        a = IndependentActors([2], learning_rate=.1, seed=7)
        for _ in range(300):
            actions = a.sample(); a.update(actions, [float(actions[0] == 1)])
        self.assertGreater(a.probabilities()[0][1], .85)
        snapshot = a.freeze()
        with self.assertRaises(RuntimeError): a.update([0], [1])
        self.assertEqual(snapshot, a.snapshot())

    def test_atomic_validation_and_no_unbounded_reward(self):
        a = IndependentActors([2, 2]); before = a.snapshot()
        with self.assertRaises(ValueError): a.update([0, 7], [1, 1])
        self.assertEqual(before, a.snapshot())
        with self.assertRaises(ValueError): a.update([0, 0], [1, float('nan')])
        self.assertEqual(before, a.snapshot())

    def test_empirical_regret_is_actual_unilateral_gain(self):
        table = {(a, b): [int(a == b), int(a == b)] for a in range(2) for b in range(2)}
        self.assertEqual(empirical_deviation(table, [[1, 0], [1, 0]]), [0, 0])
        self.assertEqual(empirical_deviation(table, [[1, 0], [0, 1]]), [1, 1])

    def test_missing_payoff_not_silently_imputed(self):
        with self.assertRaises(ValueError): empirical_deviation({(0, 0): [1, 1]}, [[.5, .5], [.5, .5]])


if __name__ == '__main__': unittest.main()
