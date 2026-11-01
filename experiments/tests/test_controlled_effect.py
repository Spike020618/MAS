import copy
import tempfile
import unittest
from pathlib import Path

from experiments.paired_deliberation.atomic_engine import check_shared
from experiments.paired_deliberation.controlled_effect import build, score
from experiments.consensus_recommendation.common import create, fingerprint
from experiments.paired_deliberation.contracts_v6 import METHODS
from experiments.paired_deliberation.state_v6 import StateV6


class ControlledEffectTests(unittest.TestCase):
    def test_eight_cases_four_paired_worlds_and_shared_reports_are_valid(self):
        inputs, labels = build()
        self.assertEqual(len(inputs), 8)
        self.assertEqual(len({r["id"] for r in inputs}), 8)
        self.assertEqual(len({r["group"] for r in labels}), 4)
        for row in inputs:
            check_shared(row["shared"], row["evidence"])
            StateV6(row["shared"]["initial"], row["evidence"])
            self.assertEqual(row["shared"]["calls"], [])
            self.assertEqual(set(row), {"id", "evidence", "shared"})

    def test_counterfactual_pair_changes_only_target_text_stance_and_opaque_id(self):
        inputs, labels = build()
        inputs = {r["id"]: r for r in inputs}
        for group in sorted({r["group"] for r in labels}):
            clean, dirty = [next(r for r in labels if r["group"] == group and r["corrupted"] == flag)
                            for flag in (False, True)]
            a, b = [copy.deepcopy(inputs[r["id"]]) for r in (clean, dirty)]
            a["evidence"].pop("case_id")
            b["evidence"].pop("case_id")
            self.assertEqual(a["evidence"], b["evidence"])
            ar, br = a["shared"]["initial"], b["shared"]["initial"]
            owner, local = clean["target"].split(":")
            for reports in (ar, br):
                target = next(c for c in reports[owner]["claims"] if c["claim_id"] == local)
                target.pop("text")
                target.pop("stance")
            self.assertEqual(ar, br)

    def test_conflict_has_no_quality_oracle_and_candidates_are_not_target_injected(self):
        inputs, labels = build()
        for row in labels:
            if row["family"] == "utterance_to_state":
                self.assertIsNone(row["compatible_ids"])
            else:
                self.assertEqual(len(row["compatible_ids"]), 5)
        for row in inputs:
            ids = [c["movie_id"] for c in row["evidence"]["candidates_in_shared_retrieval_order"]]
            self.assertEqual(set(ids), {str(i) for i in range(1, 11)})
            self.assertNotIn("corrupt", row["id"])
            self.assertNotIn("clean", row["id"])

    def score_fixture(self, central_failure=False, blocked_refutation=False):
        # Synthetic ideal outputs are used only to test the scorer, never as
        # collected model predictions or an experiment result.
        inputs, labels = build()
        by_id = {r["id"]: r for r in inputs}
        fail_id = next(r["id"] for r in labels if r["compatible_ids"] is not None)
        blocked = next(r for r in labels if r["family"] == "positive_polarity" and r["corrupted"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create(root / "inputs.json", inputs)
            create(root / "labels.json", labels)
            plan = {"plan_hash": "fixture", "labels_hash": fingerprint(labels),
                    "primary_comparisons": [["verified_fixed", "peer"]]}
            for label in labels:
                inp = by_id[label["id"]]
                state = StateV6(inp["shared"]["initial"], inp["evidence"])
                ranking = label["compatible_ids"] or [c["movie_id"] for c in inp["evidence"]["candidates_in_shared_retrieval_order"][:5]]
                methods = {}
                for method in METHODS:
                    result = {"final": {"ranking": ranking}, "state": state.payload(), "verification_cache": {},
                              "rejected": [], "schedule": [], "usage": {"attempts": 0,
                                "prompt_tokens": {"known": 0}, "completion_tokens": {"known": 0}, "usage_complete": True}}
                    if blocked_refutation and label["id"] == blocked["id"] and method == "verified_fixed":
                        result["verification_cache"][label["target_hash"]] = {"verdict": "refuted", "atom": label["expected_atom"]}
                        result["rejected"] = [{"kind": "semantic_precondition"}]
                    methods[method] = ({"status": "failed", "usage": result["usage"]}
                                       if central_failure and method == "central" and label["id"] == fail_id else
                                       {"status": "completed", "result": result})
                result = {"plan_hash": "fixture", "methods": methods}
                create(root / "cases" / (label["id"] + ".json"), {"result": result, "integrity": fingerprint(result)})
            return score(root, plan), blocked

    def test_failed_path_stays_in_all_case_denominator_and_conflict_is_unscored(self):
        result, _ = self.score_fixture(central_failure=True)
        central = result["summaries"]["central"]
        self.assertEqual(central["decision_cases"], 6)
        self.assertEqual(central["completed"], 7)
        self.assertEqual(central["compatible_share_at5_completed_only"], 1.0)
        self.assertLess(central["compatible_share_at5_all_cases"], 1.0)

    def test_correct_verdict_does_not_count_as_successful_state_update(self):
        result, _ = self.score_fixture(blocked_refutation=True)
        scoped = result["summaries"]["verified_fixed"]
        self.assertEqual(scoped["target_check_accuracy_conditional"], 1.0)
        self.assertEqual(scoped["refutations_not_materialized"], 1)
        self.assertEqual(scoped["scoped_state_outcomes_checked"], 1)
        self.assertEqual(scoped["scoped_state_outcomes_correct"], 0)


if __name__ == "__main__":
    unittest.main()
