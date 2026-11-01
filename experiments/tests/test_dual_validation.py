import copy
import json
import unittest
from collections import Counter

from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.controlled_update_gate import build as old_build
from experiments.paired_deliberation.state_v6 import StateV6
from experiments.paired_deliberation.validate_dual_evidence import build, infer


class DualValidationTests(unittest.TestCase):
    def test_frozen_instance_counts_balance_and_new_claims(self):
        rows = build()
        self.assertEqual(len(rows), 81)
        self.assertEqual(len({r["id"] for r in rows}), 81)
        self.assertEqual(Counter(r["label"] for r in rows), {"supported": 27, "refuted": 27, "insufficient": 27})
        self.assertEqual(len({r["group"] for r in rows}), 24)
        texts = {r["input"]["reports"]["current_request"]["claims"][0]["text"] for r in rows}
        prior = {r["input"]["reports"]["current_request"]["claims"][0]["text"] for r in old_build()}
        self.assertFalse(texts & prior)
        for r in rows:
            StateV6(r["input"]["reports"], r["input"]["evidence"])

    def test_conflict_scope_is_balanced_and_runtime_has_no_labels(self):
        rows = [r for r in build() if r["category"] == "unresolved_conflict"]
        self.assertEqual(len(rows), 18)
        self.assertEqual(Counter(r["scope"] for r in rows), {k: 3 for k in (
            "unresolved_positive", "unresolved_negative", "utterance_positive", "utterance_negative",
            "agreement_positive", "agreement_negative")})
        for r in rows:
            self.assertEqual(set(r["input"]), {"evidence", "reports"})
            self.assertEqual(set(r["input"]["evidence"]), {"case_id", "observations",
                "previous_mentions_excluded_not_asserted_seen", "candidates_in_shared_retrieval_order"})

    def test_new_judge_defer_preserves_sentinel_while_legacy_can_certify_one_side(self):
        row = next(r for r in build() if r["scope"] == "unresolved_negative")
        inp = {"id": row["id"], **copy.deepcopy(row["input"])}
        class Provider:
            def complete(self, slot, request):
                payload = json.loads(request["messages"][1]["content"])
                evidence = list(payload["evidence"].items())
                a, b = [{"source_id": sid, "quote": obs["text"]} for sid, obs in evidence]
                report = ({"support_spans": [b], "refute_spans": [a], "reason": "Unresolved conflict."}
                          if request["kind"] == "dual_evidence_verification" else
                          {"verdict": "supported", "evidence_spans": [b], "reason": "One negative statement."})
                return {"report": report, "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                        "model": "fixture", "elapsed_ms": 1}
        limits = {"max_calls": 2, "per_call_prompt_tokens": 8000, "per_call_completion_tokens": 1280,
                  "total_prompt_tokens": 16000, "total_completion_tokens": 2560}
        client = AttemptClient(Provider(), limits)
        dual = infer(inp, "dual", client)
        legacy = infer(inp, "legacy", client)
        self.assertEqual(dual["action"], "defer")
        self.assertEqual(legacy["action"], "keep")
        self.assertTrue(dual["sentinel_preserved"] and legacy["sentinel_preserved"])
        self.assertEqual(client.summary()["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
