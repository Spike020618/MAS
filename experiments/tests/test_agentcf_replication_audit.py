import unittest

from experiments.replications.agentcf.audit import SNAPSHOT, run, training_probe, memory_update_probe, inference_client_probe, missing_rank_probe


class AgentCFUpstreamAuditTests(unittest.TestCase):
    def test_pinned_files_and_prefixes(self):
        report = run()
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(report["verified_files"], 58)
        for data in report["datasets"].values():
            self.assertEqual(data["prefix_transition_errors"], [])
            self.assertEqual(data["catalog_missing_interaction_ids"], [])
            self.assertEqual(data["users"]["test"], 100)

    def test_original_does_not_stop_correct_samples(self):
        path = SNAPSHOT / "upstream/agentcf/model/agentcf.py"
        events = training_probe(path, [[1], [1]])
        self.assertEqual(sum(e["event"] == "forward" for e in events), 2)

    def test_original_can_reconsider_previously_correct_sample(self):
        path = SNAPSHOT / "upstream/agentcf/model/agentcf.py"
        events = training_probe(path, [[1], [0]])
        self.assertIn({"event": "wrong_feedback", "users": [1]}, events)

    def test_offline_audit_does_not_claim_performance_replication(self):
        self.assertIn("not_replication_complete", run()["status"])

    def test_default_disables_negative_item_memory_update(self):
        state = memory_update_probe(SNAPSHOT / "upstream/agentcf/model/agentcf.py", None)
        self.assertEqual(state["user"], ["initial", "user updated"])
        self.assertEqual(state["positive"], ["initial", "positive updated"])
        self.assertEqual(state["negative"], ["initial"])

    def test_explicit_negative_update_uses_original_branch(self):
        state = memory_update_probe(SNAPSHOT / "upstream/agentcf/model/agentcf.py", True)
        self.assertEqual(state["negative"], ["initial", "negative updated"])

    def test_original_evaluation_routes_to_training_temperature_client(self):
        result = inference_client_probe(SNAPSHOT / "upstream/agentcf")
        self.assertEqual(result["declared_test_temperature"], 0)
        self.assertEqual(result["actual_client_calls"][0]["temperature"], 0.2)
        self.assertEqual(result["actual_client_calls"][0]["model"], "gpt-3.5-turbo-16k-0613")

    def test_original_parser_missing_rank_prefix_is_not_handled(self):
        result = missing_rank_probe(SNAPSHOT / "upstream/agentcf")
        self.assertEqual(result["exception"], "UnboundLocalError")


if __name__ == "__main__":
    unittest.main()
