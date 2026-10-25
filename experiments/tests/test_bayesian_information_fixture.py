from fractions import Fraction as F
from itertools import product
import unittest

from experiments.core.mechanism.bayesian_information_fixture import (
    score, posterior, outcomes, expected, normal_form, equilibria,
    full_strategy_regret, correlated_duplicate_counterexample,
)


class BayesianInformationFixtureTest(unittest.TestCase):
    qualities = (F(4, 5), F(4, 5), F(4, 5))
    costs = (F(3, 100),)*3

    def test_generating_distribution_sums_exactly_one(self):
        for actions in product((0, 1), repeat=3):
            self.assertEqual(sum(p for _, _, p in outcomes(self.qualities, actions)), 1)

    def test_one_signal_bayes_and_cancellation(self):
        self.assertEqual(posterior(self.qualities, (1, 0, 0)), F(4, 5))
        self.assertEqual(posterior(self.qualities, (1, -1, 0)), F(1, 2))

    def test_exact_acquisition_potential_identity(self):
        table = normal_form(self.qualities, self.costs)
        for a, row in table.items():
            for i in range(3):
                changed = tuple(1-v if j == i else v for j, v in enumerate(a))
                self.assertEqual(table[changed]['expected_utilities'][i]-row['expected_utilities'][i],
                                 table[changed]['potential']-row['potential'])

    def test_truthful_reporting_best_reply_given_honest_peers(self):
        for a in product((0, 1), repeat=3):
            truthful = expected(self.qualities, self.costs, a)
            for i in range(3):
                if not a[i]: continue
                for mapping in product((-1, 0, 1), repeat=2):
                    changed = expected(self.qualities, self.costs, a, deviation=(i, mapping))
                    self.assertLessEqual(changed['expected_utilities'][i], truthful['expected_utilities'][i])

    def test_acquisition_equilibrium_survives_joint_action_report_deviation(self):
        table = normal_form(self.qualities, self.costs)
        ne = equilibria(table)
        self.assertTrue(ne)
        for a in ne: self.assertEqual(full_strategy_regret(self.qualities, self.costs, a), 0)

    def test_real_information_externality_not_arbitrary_action_bonus(self):
        table = normal_form(self.qualities, self.costs)
        first = table[(1, 0, 0)]['expected_collective_score']-table[(0, 0, 0)]['expected_collective_score']
        third = table[(1, 1, 1)]['expected_collective_score']-table[(0, 1, 1)]['expected_collective_score']
        self.assertNotEqual(first, third)

    def test_delayed_outcome_payments_bound_including_any_signal_lie(self):
        for a in product((0, 1), repeat=3):
            for mapping in product((-1, 0, 1), repeat=2):
                row = expected(self.qualities, self.costs, a, deviation=(0, mapping))
                self.assertTrue(all(0 <= p <= 1 for p in row['expected_payments']))

    def test_brier_value_of_information_identity_requires_true_posteriors(self):
        expected_improvement = F(0)
        squared_change = F(0)
        for y, signals, mass in outcomes(self.qualities, (1, 0, 0)):
            q = posterior(self.qualities, signals)
            expected_improvement += mass*(score(q, y)-score(F(1, 2), y))
            squared_change += mass*(q-F(1, 2))**2
        self.assertEqual(expected_improvement, squared_change)

    def test_correlated_evidence_must_not_be_counted_independent(self):
        r = correlated_duplicate_counterexample()
        self.assertEqual(r['true_posterior'], F(4, 5))
        self.assertGreater(r['loss'], 0)
        self.assertGreater(r['wrong_independence_posterior'], r['true_posterior'])

    def test_zero_reward_no_resource_purchase(self):
        table = normal_form(self.qualities, self.costs, F(0))
        self.assertEqual(equilibria(table), [(0, 0, 0)])


if __name__ == '__main__': unittest.main()
