import itertools
import unittest

from experiments.paired_deliberation.native_quality_review_game import QualityReviewGame


class QualityReviewGameTests(unittest.TestCase):
    def test_exact_potential_identity_exhaustive(self):
        game = QualityReviewGame([[.2, -.1], [.4, .1], [.1, .3]],
                                 [[0., .02], [.01, 0.], [.03, .01]], cost_weight=2.)
        for state in itertools.product(game.actions, repeat=3):
            for i in range(3):
                for other in game.deviations(state, i):
                    self.assertAlmostEqual(game.utility(other, i) - game.utility(state, i),
                                           game.potential(other) - game.potential(state), places=12)

    def test_central_enumeration_and_equilibrium(self):
        game = QualityReviewGame([[.3, .29], [.28, .01], [.02, .1]], [[.01]*2]*3)
        result = game.solve()
        exhaustive = max(game.potential(s) for s in itertools.product(game.actions, repeat=3))
        self.assertAlmostEqual(result['central_potential'], exhaustive)
        self.assertGreaterEqual(result['central_gap'], 0.)
        self.assertTrue(result['epsilon_equilibrium'])
        self.assertLessEqual(result['max_feasible_unilateral_gain'], 1e-10)
        for move in result['trace']:
            self.assertAlmostEqual(move['potential_gain'], move['utility_gain'])

    def test_negative_and_zero_quality_do_not_purchase_reviews(self):
        game = QualityReviewGame([[-.1, 0.], [-.5, -.2], [0., -.1]], [[0.]*2]*3)
        self.assertEqual(game.solve()['actions'], (None, None, None))

    def test_real_zero_cost_and_redundancy_cleanup_terminate(self):
        game = QualityReviewGame([[.3], [.3], [.1]], [[0.], [0.], [0.]])
        result = game.solve(initial=(0, 0, 0))
        self.assertEqual(result['cost_cny'], 0.)
        self.assertEqual(sum(x is not None for x in result['actions']), 1)
        self.assertTrue(any(move['zero_gain_abstention_cleanup'] for move in result['trace']))
        self.assertTrue(result['epsilon_equilibrium'])
        self.assertLessEqual(result['visited_states'], 8)

    def test_duplicate_coverage_is_maximum_not_sum(self):
        game = QualityReviewGame([[.5], [.4], [.3]], [[.01], [.01], [.01]])
        self.assertEqual(game.coverage((0, 0, 0)), .5)
        self.assertAlmostEqual(game.utility((0, 0, 0), 1), -.01)
        self.assertAlmostEqual(game.utility((0, None, None), 0), .49)

    def test_shared_budget_feasible_deviation_gne(self):
        game = QualityReviewGame([[1., 0.], [0., 2.], [.1, .1]],
                                 [[.8]*2, [.8]*2, [.8]*2], budget=.8)
        result = game.solve()
        self.assertEqual(result['equilibrium_kind'], 'generalized_nash')
        self.assertLessEqual(result['cost_cny'], .8)
        self.assertTrue(result['epsilon_equilibrium'])
        # A local feasible equilibrium need not be the central optimum:
        # a second player cannot evict the current budget occupant unilaterally.
        self.assertGreater(result['central_gap'], 0.)
        self.assertAlmostEqual(result['central_potential'], 1.2)
        for move in result['trace']:
            self.assertTrue(game.feasible(move['to']))
        with self.assertRaises(ValueError):
            game.solve(initial=(0, 1, None))

    def test_constraint_applies_to_central_comparator(self):
        game = QualityReviewGame([[5., 1.], [1., 5.]], [[2., 1.], [1., 2.]], budget=1.)
        self.assertAlmostEqual(game.potential(game.central()), 0.)
        self.assertTrue(game.feasible(game.central()))

    def test_invalid_inputs(self):
        for gains, costs in [([], []), ([[1.]], [[-1.]]), ([[float('nan')]], [[0.]]),
                             ([[1., 2.]], [[0.]]), ([[1.]], [[float('inf')]])]:
            with self.assertRaises(ValueError):
                QualityReviewGame(gains, costs)
        game = QualityReviewGame([[1.]], [[0.]])
        for state in [(True,), (1,), (), (-1,)]:
            with self.assertRaises(ValueError):
                game.solve(state)


if __name__ == '__main__':
    unittest.main()
