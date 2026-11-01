import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.paired_deliberation.calibration_v51 import build, export
from experiments.paired_deliberation.client_v5 import BoundedClient
from experiments.paired_deliberation.comparison_v51 import generate_shared, run_method, run_suite, METHODS
from experiments.paired_deliberation.run_v51 import manifest, score_calibration
from experiments.consensus_recommendation.common import create


class FixtureProvider:
    def __init__(self):
        self.calls = []

    def complete(self, slot, request):
        self.calls.append((slot, copy.deepcopy(request)))
        payload = json.loads(request["messages"][1]["content"])
        if request["kind"] == "initial":
            focus = payload["task"]
            text = ("Movie 1 has catalog year 2001." if focus.startswith("Interpret")
                    else "Movie 1 has catalog year 2000." if focus.startswith("Assess")
                    else "The seeker likes Drama.")
            refs = ["message:1"] if text.startswith("The") else ["movie:1"]
            report = {"ranking": [str(i) for i in range(1, 6)], "updates": [],
                      "claims": [{"claim_id": "c1", "text": text, "source_ids": refs,
                                  "candidate_ids": ["1"], "stance": "support"}]}
        elif request["kind"] == "verification":
            report = {"verdict": "insufficient", "reason": "No evidence of liking.", "evidence_spans": []}
        elif request["kind"] == "whole_review":
            report = {"ranking": [str(i) for i in range(1, 6)], "updates": [],
                      "claims": [copy.deepcopy(payload["target"]["claim"])]}
        elif request["kind"] == "scoped_review":
            report = {"target_claim_id": payload["target_claim_id"],
                      "expected_hash": payload["schema"]["expected_hash"],
                      "requested_action": "keep", "replacement": None}
        elif request["kind"] == "ranking":
            report = {"ranking": [str(i) for i in range(1, 6)]}
        else:
            raise AssertionError(request["kind"])
        return {"report": report, "model": "fixture", "elapsed_ms": 1,
                "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                          "tool_calls": 0, "tool_units": 0, "retries": 0, "actual_cost": None}}


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.evidence = {"case_id": "fixture:51", "observations": [
            {"source_id": "message:1", "role": "seeker", "text": "Tell me about a film."}],
            "previous_mentions_excluded_not_asserted_seen": [],
            "candidates_in_shared_retrieval_order": [
                {"source_id": "movie:" + str(i), "movie_id": str(i), "title": "Film " + str(i),
                 "year": 2000, "genres": ["Drama"]} for i in range(1, 6)]}
        self.limits = {"per_call_prompt_tokens": 100000, "per_call_completion_tokens": 2000,
                       "total_prompt_tokens": 1000000, "total_completion_tokens": 100000}
        self.provider = FixtureProvider()
        self.shared = generate_shared(self.evidence, BoundedClient(self.provider, self.limits))

    def client(self, _method=None):
        return BoundedClient(self.provider, self.limits)

    def test_central_and_proposed_receive_identical_feedback_and_complete_costs(self):
        central = run_method(self.evidence, self.shared, "central_matched", self.client())
        proposed = run_method(self.evidence, self.shared, "verified_scoped", self.client())
        self.assertEqual(central["shared_feedback_hash"], proposed["shared_feedback_hash"])
        self.assertEqual(central["usage"]["calls"], proposed["usage"]["calls"])
        self.assertEqual(central["usage"]["calls"], 8)
        self.assertEqual(proposed["usage"]["prompt_tokens"], 40)
        review_payloads = [json.loads(req["messages"][1]["content"]) for _, req in self.provider.calls
                           if req["kind"] in {"whole_review", "scoped_review"}]
        self.assertTrue(all(row["evidence"] == self.evidence for row in review_payloads))
        central_history_sizes = [len(row["controller_history"]) for row in review_payloads if row["reviewer"] == "central"]
        self.assertEqual(central_history_sizes, [0, 1, 2])

    def test_no_runtime_verifier_cannot_see_feedback_and_leaves_wrong_claim_unverified(self):
        start = len(self.provider.calls)
        result = run_method(self.evidence, self.shared, "no_runtime_verifier", self.client())
        self.assertFalse(result["uses_shared_feedback"])
        self.assertEqual(result["usage"]["calls"], 7)
        self.assertIn("current_request:c1", result["claim_state"]["uncertain_claims"])
        for _, request in self.provider.calls[start:]:
            self.assertNotEqual(request["kind"], "verification")
            payload = json.loads(request["messages"][1]["content"])
            self.assertFalse(payload.get("verification_feedback", payload.get("shared_verification_feedback")))

    def test_verifier_removal_and_forced_binary_change_actual_state(self):
        full = run_method(self.evidence, self.shared, "verified_scoped", self.client())
        binary = run_method(self.evidence, self.shared, "force_binary", self.client())
        self.assertIn("current_request:c1", full["claim_state"]["retracted_ids"])
        self.assertIn("movie_fit:c1", full["claim_state"]["uncertain_claims"])
        forced = binary["claim_state"]["supported_claims"]["movie_fit:c1"]
        self.assertTrue(forced["verification"]["forced_from_insufficient"])

    def test_all_methods_execute_and_scope_ablation_alias_has_no_extra_calls(self):
        result = run_suite(self.evidence, self.shared, self.client)
        self.assertEqual(set(result), set(METHODS) | {"no_scope_constraint"})
        self.assertEqual(result["no_scope_constraint"]["alias_of"], "peer_matched")
        self.assertEqual(result["no_scope_constraint"]["extra_calls"], 0)
        self.assertTrue(all(not result[name]["rejected_proposals"] for name in METHODS))

    def test_bundle_tampering_and_duplicate_shared_costs_are_rejected(self):
        changed = copy.deepcopy(self.shared)
        changed["feedback"]["current_request:c1"]["verification"]["verdict"] = "supported"
        with self.assertRaisesRegex(ValueError, "integrity"):
            run_method(self.evidence, changed, "verified_scoped", self.client())
        client = self.client()
        client.account_shared(**self.shared["initial_calls"][0])
        with self.assertRaisesRegex(ValueError, "already charged"):
            client.account_shared(**self.shared["initial_calls"][0])

    def test_whole_review_can_change_scope_but_scoped_review_rejects_extra_fields(self):
        original = self.provider.complete
        def malicious(slot, request):
            record = original(slot, request)
            if request["kind"] == "whole_review":
                record["report"]["claims"][0]["claim_id"] = "rewritten"
            if request["kind"] == "scoped_review":
                record["report"]["ranking"] = ["5", "4", "3", "2", "1"]
            return record
        self.provider.complete = malicious
        ordinary = run_method(self.evidence, self.shared, "peer_matched", self.client())
        constrained = run_method(self.evidence, self.shared, "verified_scoped", self.client())
        self.assertTrue(any(key.endswith(":rewritten") for key in ordinary["claim_state"]["uncertain_claims"]))
        self.assertEqual(len(constrained["rejected_proposals"]), 3)
        self.assertEqual(constrained["events"], [])

    def test_false_support_blocks_gate_despite_high_overall_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = manifest(root)
            rows = json.loads((root / "calibration/labels.json").read_text())
            changed = False
            for row in rows:
                prediction = row["label"]
                if not changed and row["label"] == "insufficient":
                    prediction, changed = "supported", True
                create(root / "calibration/predictions" / (row["id"] + ".json"), {"verdict": prediction})
            result = score_calibration(root, plan["calibration"])
            self.assertGreater(result["rates"]["all"], .98)
            self.assertEqual(result["false_supported"], 1)
            self.assertFalse(result["pass"])

    def test_calibration_inputs_are_blind_balanced_and_deterministic(self):
        rows = build()
        self.assertEqual(rows, build())
        self.assertEqual(len(rows), 84)
        self.assertEqual(len({row["id"] for row in rows}), 84)
        self.assertEqual(Counter(row["label"] for row in rows),
                         {"supported": 28, "refuted": 28, "insufficient": 28})
        self.assertEqual(len({row["category"] for row in rows}), 7)
        with tempfile.TemporaryDirectory() as directory:
            manifest = export(directory)
            inputs = json.loads((Path(directory) / "inputs.json").read_text())
        self.assertEqual(manifest["items"], 84)
        self.assertTrue(all(set(row) == {"id", "claim", "evidence"} for row in inputs))


if __name__ == "__main__":
    unittest.main()
