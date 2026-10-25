import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "experiments/mechanism/estimate_follower_response.py"
SPEC = importlib.util.spec_from_file_location("estimate_follower_response", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EmpiricalStackelbergTests(unittest.TestCase):
    def test_parse_contract_defaults_unit_cost(self):
        self.assertEqual(
            MODULE.parse_contract("independent__static_aeic@0.2"),
            (0.2, 1.0),
        )
        self.assertEqual(
            MODULE.parse_contract("independent__static_aeic@0.5__cost@1.5"),
            (0.5, 1.5),
        )

    def test_balanced_selection_uses_frozen_utility(self):
        grid = [
            {"contract_id": "cheap", "accuracy": 0.80,
             "mean_tokens_per_agent_round": 100.0,
             "mean_execution_calls_per_agent_round": 1.0, "failure_rate": 0.0},
            {"contract_id": "expensive", "accuracy": 0.82,
             "mean_tokens_per_agent_round": 1000.0,
             "mean_execution_calls_per_agent_round": 2.0, "failure_rate": 0.0},
        ]
        presets = {"balanced": {"token": 0.15, "call": 0.10, "failure": 0.10}}
        result = MODULE.select_contracts(grid, presets)
        self.assertEqual(result["balanced"]["selected"]["contract_id"], "cheap")

    def test_pareto_frontier_removes_dominated_contract(self):
        grid = [
            {"contract_id": "a", "accuracy": 0.80,
             "mean_tokens_per_agent_round": 100.0,
             "mean_execution_calls_per_agent_round": 1.0},
            {"contract_id": "b", "accuracy": 0.79,
             "mean_tokens_per_agent_round": 110.0,
             "mean_execution_calls_per_agent_round": 1.0},
        ]
        self.assertEqual(MODULE.pareto_frontier(grid), ["a"])

    def test_summary_reports_parallel_round_latency(self):
        rounds = [
            {"round_id": 1, "chosen_action": "aeic_effort", "token_input": 10,
             "token_output": 5, "latency_s": 2.0, "action_trace": [{"stage": "aeic_effort"}]},
            {"round_id": 1, "chosen_action": "low_effort", "token_input": 8,
             "token_output": 2, "latency_s": 3.0, "action_trace": [{"stage": "low_effort"}]},
            {"round_id": 2, "chosen_action": "verify", "token_input": 12,
             "token_output": 6, "latency_s": 4.0, "action_trace": [{"stage": "verify_draft"}, {"stage": "verify_revision"}]},
        ]
        record = {"completed": True, "error": None, "condition": "independent__static_aeic@0.2",
                  "final_decision": "yes", "ground_truth": "yes", "rounds": rounds}
        result = MODULE.summarize([record])[0]
        self.assertEqual(result["mean_estimated_question_latency_s"], 7.0)
        self.assertAlmostEqual(result["mean_latency_s_per_agent_round"], 3.0)


if __name__ == "__main__":
    unittest.main()
