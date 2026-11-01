import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.dual_runtime import DualRuntimeClient, _run_suite, run_suite
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.runner_v6 import shared_initial
from experiments.paired_deliberation.verified_state import EvidenceVerifier, VerificationError
from experiments.tests.test_design_v6 import Provider, LIMITS, evidence


class RuntimeTests(unittest.TestCase):
    def test_actual_dual_request_is_charged_and_source_record_not_mutated(self):
        class P:
            def complete(self, slot, request):
                self.kind = request["kind"]
                self.report = {"support_spans": [{"source_id": "m", "quote": "I want Drama."}],
                               "refute_spans": [{"source_id": "n", "quote": "I do not want Drama."}],
                               "reason": "Unresolved conflict."}
                return {"report": copy.deepcopy(self.report), "model": "fixture", "elapsed_ms": 1,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        p = P()
        base = AttemptClient(p, LIMITS)
        client = DualRuntimeClient(base)
        sources = {"m": {"text": "I want Drama."}, "n": {"text": "I do not want Drama."}}
        claim = {"text": "The seeker wants Drama.", "source_ids": list(sources)}
        def judge(payload):
            record = client.complete("slot", RequestsV5.verification(payload))
            return {**record["report"], "usage": record["usage"]}
        result = EvidenceVerifier(sources, judge).verify(claim)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertEqual(p.kind, "dual_evidence_verification")
        self.assertNotIn("verdict", p.report)
        self.assertEqual(base.records[0]["kind"], "dual_evidence_verification")
        self.assertEqual(client.summary()["attempts"], 1)
        self.assertTrue(client.conversions[0]["verification"]["evidence_conflict"])

    def test_invalid_dual_response_remains_a_failure_after_accounting(self):
        class P:
            def complete(self, slot, request):
                return {"report": {"verdict": "supported"}, "model": "fixture", "elapsed_ms": 1,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        client = DualRuntimeClient(AttemptClient(P(), LIMITS))
        with self.assertRaises(VerificationError):
            client.complete("slot", RequestsV5.verification({"claim": {"text": "A", "source_ids": ["m"]},
                                                          "evidence": {"m": {"text": "A"}}}))
        self.assertEqual(client.summary()["attempts"], 1)
        self.assertEqual(client.conversions[0]["status"], "invalid_dual_response")

    def test_five_conditions_still_run_through_same_engine(self):
        e = evidence()
        provider = Provider()
        initial = shared_initial(e, AttemptClient(provider, LIMITS), "shared")
        result = _run_suite(e, lambda _: AttemptClient(provider, LIMITS), initial)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(r["status"] == "completed" for r in result.values()))
        self.assertIn("current_request:c1", result["verified_fixed"]["result"]["state"]["retracted_ids"])

    def test_failed_semantic_release_blocks_before_any_condition_or_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "release_score.json").write_text(json.dumps({"pass": False, "gates": {"coverage": False}}))
            def forbidden(_method):
                raise AssertionError("must not create a client")
            with self.assertRaisesRegex(RuntimeError, "semantic gate failed"):
                run_suite(evidence(), forbidden, directory)


if __name__ == "__main__":
    unittest.main()
