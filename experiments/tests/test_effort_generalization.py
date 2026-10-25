import unittest
from experiments.mechanism.audit_effort_generalization import (
    prospective_hoeffding_radius, peer_influence_counterexample,
)


class EffortGeneralizationTest(unittest.TestCase):
    def test_more_samples_reduce_prospective_radius_not_zero_empirical_regret(self):
        r = prospective_hoeffding_radius(3000, 24)
        self.assertGreater(r, 0)
        self.assertAlmostEqual(prospective_hoeffding_radius(3000, 96), r/2)

    def test_uniform_multiple_comparisons_cost_is_not_ignored(self):
        self.assertGreater(prospective_hoeffding_radius(3000, 24, family=96),
                           prospective_hoeffding_radius(3000, 24, family=3))

    def test_invalid_statistical_design_fails_closed(self):
        for kwargs in ({'n': 0}, {'delta': 0}, {'family': 0}, {'high_cost_cap': -1}):
            parameters = dict(scale=3000, n=24)
            parameters.update(kwargs)
            with self.assertRaises(ValueError): prospective_hoeffding_radius(**parameters)

    def test_final_vote_removal_does_not_remove_causal_peer_influence(self):
        c = peer_influence_counterexample()
        self.assertEqual(c['agent_zero_credit_before'], 0)
        self.assertEqual(c['agent_zero_credit_after'], 0)
        self.assertEqual(c['own_utility_change'], -10)
        self.assertEqual(c['collective_potential_change'], 990)
        self.assertEqual(c['identity_error'], 1000)


if __name__ == '__main__': unittest.main()
