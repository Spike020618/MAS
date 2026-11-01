import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.information_roles_pilot.core import messages, rrf, validate_rank
from experiments.information_roles_pilot.evaluate import score
from experiments.information_roles_pilot.run import Collector


def fixture_case():
    return {"case_id": "fixture", "user_id": "PRIVATE_LOCAL_ID", "positive_ids": ["SECRET_LABEL"],
            "candidate_ids": ["1", "2"], "P0": ["1", "2"], "C0": ["2", "1"],
            "catalog": [{"movie_id": "1", "title": "Film A"}, {"movie_id": "2", "title": "Film B"}],
            "history_packet": [{"record_id": "h0", "observed_rating": 4}],
            "cf_packet": [{"movie_id": "1", "predicted_rating": 3}]}


class InformationRolePilotTests(unittest.TestCase):
    def test_labels_and_identity_cannot_enter_prompt(self):
        for role in ("P", "C", "U1", "U2"):
            text = json.dumps(messages(fixture_case(), role))
            self.assertNotIn("PRIVATE_LOCAL_ID", text)
            self.assertNotIn("SECRET_LABEL", text)
            self.assertNotIn("positive_ids", text)

    def test_same_union_information_for_unified_control(self):
        union = messages(fixture_case(), "U1")[1]["content"]
        self.assertIn("observed_history", union)
        self.assertIn("collaborative_predictions", union)
        self.assertNotIn("collaborative_predictions", messages(fixture_case(), "P")[1]["content"])

    def test_complete_permutation_required(self):
        for bad in (["1"], ["1", "1"], ["1", "3"], [1, 2]):
            with self.assertRaises(ValueError):
                validate_rank(bad, ["1", "2"])

    def test_rrf_has_no_agent_order_bias(self):
        self.assertEqual(rrf(["1", "2"], ["2", "1"]), rrf(["2", "1"], ["1", "2"]))
        self.assertEqual(rrf(["2", "1"], ["2", "1"]), ["2", "1"])

    def test_end_to_end_denominator_keeps_outside_pool_targets(self):
        metric = score(["1", "2"], {"1", "999"})
        self.assertEqual(metric["hit_at_5"], 1)
        self.assertEqual(metric["recall_at_5"], 0.5)

    def test_label_permutation_does_not_change_request(self):
        case = fixture_case()
        before = messages(case, "U1")
        case["positive_ids"] = ["different label"]
        self.assertEqual(before, messages(case, "U1"))

    def make_collector(self):
        collector = Collector.__new__(Collector)
        collector.plan, collector.key = {"fixture": True}, "TEST_NOT_A_REAL_KEY"
        collector.endpoint = "https://api.deepseek.com/chat/completions"
        collector.lock, collector.models = threading.Lock(), set()
        collector.halt = threading.Event()
        collector.new_calls = collector.cached_calls = 0
        return collector

    def response(self, ranking):
        raw = {"model": "fixture-backend", "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
               "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"ranking": ranking})}}]}
        return type("Response", (), {"status_code": 200, "json": lambda _: raw})()

    def test_resume_uses_zero_new_requests(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.information_roles_pilot.run.OUT", Path(tmp)):
            collector = self.make_collector()
            with patch("requests.post", return_value=self.response(["1", "2"])) as post:
                first = collector.call(fixture_case(), "P")
                second = collector.call(fixture_case(), "P")
                self.assertEqual(post.call_count, 1)
                self.assertEqual(first, second)

    def test_invalid_output_is_not_filled_in(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.information_roles_pilot.run.OUT", Path(tmp)):
            collector = self.make_collector()
            with patch("requests.post", return_value=self.response(["1"])):
                result = collector.call(fixture_case(), "P")
                self.assertIsNone(result["ranking"])
                self.assertIsNotNone(result["error"])

    def test_unfinished_intent_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.information_roles_pilot.run.OUT", Path(tmp)):
            collector = self.make_collector()
            intent = Path(tmp) / "slots/fixture/P.intent.json"
            intent.parent.mkdir(parents=True)
            intent.write_text("{}")
            with patch("requests.post") as post, self.assertRaises(RuntimeError):
                collector.call(fixture_case(), "P")
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
