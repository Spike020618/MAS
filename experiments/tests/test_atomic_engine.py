import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.paired_deliberation.atomic_engine import run_method, run_suite
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.runner_v6 import shared_initial
from experiments.tests.test_design_v6 import Provider, LIMITS, evidence, claim, report


class AtomicEngineTests(unittest.TestCase):
    def test_uncited_opposition_is_checked_and_both_stage_calls_charged(self):
        e = evidence()
        e["observations"] = [{"source_id": "message:1", "role": "seeker", "text": "I want Drama."},
                             {"source_id": "message:2", "role": "seeker", "text": "I do not want Drama."}]
        class P(Provider):
            def complete(self, slot, req):
                if req["kind"] in {"initial_v6", "atomic_interpretation", "atomic_source_relations"}:
                    self.calls.append((slot, copy.deepcopy(req)))
                    if req["kind"] == "initial_v6":
                        c = claim(text="The seeker wants Drama.", kind="explicit_preference")
                        c["source_ids"] = ["message:1"]
                        obj = report([c])
                    elif req["kind"] == "atomic_interpretation":
                        obj = {"scope": "preference_state", "subject": "seeker", "predicate": "wants",
                               "object": "Drama", "polarity": True, "time": "current", "reason": "Preference state."}
                    else:
                        obj = {"relations": [{"source_id": "message:1", "relation": "supports", "quote": "I want Drama."},
                                             {"source_id": "message:2", "relation": "refutes", "quote": "I do not want Drama."}],
                               "reason": "Opposing sources."}
                    return {"report": obj, "model": "offline-fixture", "elapsed_ms": 1,
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
                return super().complete(slot, req)
        p = P()
        shared = shared_initial(e, AttemptClient(p, LIMITS), "shared")
        result = run_method(e, "verified_fixed", AttemptClient(p, LIMITS), "fixed", shared)
        self.assertFalse(result["state"]["supported_claims"])
        self.assertEqual(len(result["state"]["uncertain_claims"]), 3)
        self.assertEqual(result["usage"]["attempts"], 9)  # 3 initial + parse/source once + 3 reviews + output.
        check = next(iter(result["verification_cache"].values()))
        self.assertIn("message:2", check["source_ids"])
        self.assertTrue(check["evidence_conflict"])

    def test_compound_reservation_keeps_final_output_before_any_two_stage_action(self):
        e, p = evidence(), Provider()
        result = run_method(e, "verified_fixed", AttemptClient(p, {**LIMITS, "max_calls": 8}), "tight")
        self.assertEqual(result["stop"], "budget_reserved_for_final")
        self.assertFalse(result["schedule"])
        self.assertEqual(result["usage"]["attempts"], 4)

    def test_missing_or_failed_release_never_creates_a_runtime_client(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "release_score.json").write_text(json.dumps({"pass": False, "gates": {"faithfulness": False}}))
            def forbidden(method):
                raise AssertionError("no client creation before passing gate")
            with self.assertRaisesRegex(RuntimeError, "atomic semantic gate failed"):
                run_suite(evidence(), forbidden, directory)

    def test_invalid_controller_target_consumes_round_without_substitution_or_path_abort(self):
        class P(Provider):
            def complete(self, slot, request):
                r = super().complete(slot, request)
                if request["kind"] == "select_v6":
                    r["report"] = {"target_claim_id": "not_an_allowed_target"}
                return r
        p = P()
        result = run_method(evidence(), "central", AttemptClient(p, LIMITS), "central", max_rounds=2)
        self.assertEqual(result["stop"], "controller_invalid_action")
        self.assertEqual(len(result["rejected"]), 2)
        self.assertTrue(all(r["kind"] == "invalid_action" for r in result["rejected"]))
        self.assertFalse(result["events"])
        self.assertEqual(sum(req["kind"] == "select_v6" for _, req in p.calls), 2)
        self.assertFalse(any(req["kind"] == "whole_review_v6" for _, req in p.calls))
        self.assertEqual(p.calls[-1][1]["kind"], "ranking_v6")


if __name__ == "__main__":
    unittest.main()
