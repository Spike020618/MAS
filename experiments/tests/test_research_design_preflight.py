import unittest

from experiments.replications.agentcf.design_preflight import run_checks


class ResearchDesignCounterexamples(unittest.TestCase):
    def test_potential_ne_can_be_strictly_suboptimal(self):
        result = run_checks()["potential_game"]
        self.assertIn([0, 0], result["team_ne"])
        self.assertLess(result["bad_ne_value"], result["global_best_value"])

    def test_difference_reward_does_not_change_pure_ne(self):
        result = run_checks()["potential_game"]
        self.assertEqual(result["team_ne"], result["difference_ne"])
        self.assertTrue(result["unilateral_gains_identical"])

    def test_greedy_can_miss_complementary_queries(self):
        result = run_checks()["myopic_information_value"]
        self.assertTrue(all(v < 0 for v in result["single_action_gains"]))
        self.assertGreater(result["joint_gain"], 0)

    def test_role_count_is_not_source_count(self):
        result = run_checks()["duplicate_evidence"]
        self.assertEqual(result["reports"], 3)
        self.assertEqual(result["unique_sources"], 1)


if __name__ == "__main__":
    unittest.main()
