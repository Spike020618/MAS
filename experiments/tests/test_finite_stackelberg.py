import unittest

from experiments.mechanism.solve_finite_stackelberg import solve
from experiments.mechanism.audit_equilibrium import audit


class FiniteGameTests(unittest.TestCase):
    def test_follower_best_reply_precedes_leader_optimization(self):
        game = {"actions": ["low", "high"], "contracts": [
            {"id": "a", "payoffs": {"low": {"leader": 1, "follower": 2},
                                     "high": {"leader": 100, "follower": 1}}},
            {"id": "b", "payoffs": {"low": {"leader": 2, "follower": 0},
                                     "high": {"leader": 3, "follower": 1}}}]}
        result = solve(game)
        self.assertEqual(result["solutions"]["optimistic"]["contract_id"], "b")

    def test_tie_conventions_produce_different_commitments(self):
        game = {"actions": ["low", "high"], "contracts": [
            {"id": "a", "payoffs": {"low": {"leader": 0, "follower": 1},
                                     "high": {"leader": 5, "follower": 1}}},
            {"id": "b", "payoffs": {"low": {"leader": 2, "follower": 2},
                                     "high": {"leader": 3, "follower": 1}}}]}
        result = solve(game)["solutions"]
        self.assertEqual(result["optimistic"]["contract_id"], "a")
        self.assertEqual(result["pessimistic"]["contract_id"], "b")

    def test_missing_counterfactual_is_rejected(self):
        with self.assertRaises(ValueError):
            solve({"actions": ["low", "high"], "contracts": [
                {"id": "a", "payoffs": {"low": {"leader": 1, "follower": 1}}}]})

    def test_negative_upper_bound_rejects_best_response(self):
        result = audit([{"completed": True, "condition": "selected", "rounds": [
            {"chosen_action": "aeic_effort", "offered": {
                "credit_rule": "fixed_process_quality",
                "visible_reward": {"complete_aeic": 0.2, "consistency": 0.2},
                "visible_cost": {"low_effort": 0, "aeic_effort": 1.5}}}]}])
        row = result["contracts"]["selected"]
        self.assertFalse(row["necessary_best_response_check_pass"])
        self.assertAlmostEqual(row["mean_regret_lower_bound"], 1.1)


if __name__ == "__main__":
    unittest.main()
