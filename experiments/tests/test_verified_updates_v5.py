import copy
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.client_v5 import BoundedClient
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.pipeline_v5 import run_case
from experiments.paired_deliberation.requests_v5 import RequestsV5
from experiments.paired_deliberation.verified_state import ClaimState, EvidenceVerifier, canonical
from experiments.paired_deliberation.repair_acceptance_v5 import run as controlled_acceptance
from experiments.paired_deliberation.run_v5 import run as prepare_v5
from experiments.paired_deliberation.client_v5 import RawDeepSeek, parse_raw


def claim(text="Movie 1 has catalog year 2001.", local="c1"):
    return {"claim_id": local, "text": text, "source_ids": ["movie:1"],
            "candidate_ids": ["1"], "stance": "support"}


class VerifiedUpdatesTests(unittest.TestCase):
    def setUp(self):
        self.sources = {"movie:1": {"source_id": "movie:1", "movie_id": "1",
                                   "year": 2000, "genres": ["Drama"]},
                        "message:1": {"text": "I liked that film last year."}}
        self.reports = {"A": {"ranking": ["1"], "claims": [claim(), claim("Movie 1 has catalog genre Drama.", "c2")]},
                        "B": {"ranking": ["2"], "claims": []}}
        self.state = ClaimState(self.reports, EvidenceVerifier(self.sources))

    def patch(self, replacement=None, target="A:c1"):
        return {"target_claim_id": target, "expected_hash": fingerprint(self.state.claims[target]),
                "replacement": replacement}

    def test_refutation_retracts_wrong_claim_and_removes_it_from_output_evidence(self):
        untouched = copy.deepcopy(self.state.claims["A:c2"])
        event = self.state.apply("B", self.patch())
        self.assertEqual(event["action"], "retract")
        self.assertEqual(self.state.claims["A:c1"]["status"], "retracted")
        payload = self.state.aggregation_payload()
        self.assertNotIn("A:c1", payload["supported_claims"])
        self.assertNotIn("A:c1", payload["uncertain_claims"])
        self.assertEqual(untouched, self.state.claims["A:c2"])
        self.assertNotIn("2001", canonical(payload))

    def test_supported_claim_is_preserved(self):
        event = self.state.apply("B", self.patch(target="A:c2"))
        self.assertEqual(event["action"], "keep")
        self.assertIn("A:c2", self.state.aggregation_payload()["supported_claims"])

    def test_verified_replacement_is_scoped_and_double_checked(self):
        event = self.state.apply("B", self.patch(claim("Movie 1 has catalog year 2000.")))
        self.assertEqual(event["action"], "replace")
        self.assertEqual(event["old_verification"]["verdict"], "refuted")
        self.assertEqual(event["replacement_verification"]["verdict"], "supported")
        self.assertEqual(self.reports["A"]["claims"][0]["text"], "Movie 1 has catalog year 2001.")

    def test_invalid_replacement_is_atomic(self):
        before = copy.deepcopy(self.state.claims)
        with self.assertRaises(ValueError):
            self.state.apply("B", self.patch(claim("Movie 1 has catalog year 2002.")))
        self.assertEqual(before, self.state.claims)
        self.assertEqual([], self.state.history)

    def test_cannot_smuggle_ranking_candidate_edits_or_unknown_target(self):
        patch = self.patch()
        patch["ranking"] = ["2"]
        with self.assertRaises(ValueError):
            self.state.apply("B", patch)
        replacement = claim("Movie 1 has catalog year 2000.")
        replacement["candidate_ids"] = ["2"]
        with self.assertRaises(ValueError):
            self.state.apply("B", self.patch(replacement))
        with self.assertRaises(ValueError):
            self.state.apply("B", {"target_claim_id": "A:invented", "expected_hash": "x", "replacement": None})

    def test_stale_patch_cannot_overwrite_new_revision(self):
        patch = self.patch(target="A:c2")
        self.state.apply("B", patch)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.state.apply("B", patch)

    def test_arbitrary_preference_is_unresolved_without_external_verifier(self):
        self.reports["A"]["claims"][0] = claim("The user currently wants Drama.")
        self.state = ClaimState(self.reports, EvidenceVerifier(self.sources))
        result = self.state.apply("B", self.patch())
        self.assertEqual(result["action"], "defer")
        self.assertIn("A:c1", self.state.aggregation_payload()["uncertain_claims"])

    def test_natural_language_judge_is_used_but_fake_quote_fails(self):
        item = {**claim("The user liked the film previously."), "source_ids": ["message:1"]}
        seen = []
        def judge(payload):
            seen.append(payload)
            return {"verdict": "supported", "reason": "Explicit past preference.",
                    "evidence_spans": [{"source_id": "message:1", "quote": "I liked that film"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
        self.assertEqual(EvidenceVerifier(self.sources, judge).verify(item)["kind"], "model_proxy")
        self.assertEqual(set(seen[0]), {"claim", "evidence"})
        def fake(payload):
            result = judge(payload)
            result["evidence_spans"][0]["quote"] = "I hate Drama"
            return result
        with self.assertRaisesRegex(ValueError, "fabricated"):
            EvidenceVerifier(self.sources, fake).verify(item)

    def test_runtime_invalid_quote_defers_without_becoming_supported(self):
        item = {**claim("The user currently wants Drama."), "source_ids": ["message:1"]}
        def fake(_payload):
            return {"verdict": "supported", "reason": "Unreliable judgement.",
                    "evidence_spans": [{"source_id": "message:1", "quote": "nonexistent quote"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
        state = ClaimState({"A": {"claims": [item]}, "B": {"claims": []}},
                           EvidenceVerifier(self.sources, fake), invalid_verification="defer")
        result = state.apply("B", {"target_claim_id": "A:c1",
            "expected_hash": fingerprint(state.claims["A:c1"]), "replacement": None})
        self.assertEqual(result["action"], "defer")
        self.assertTrue(result["old_verification"]["verification_failed"])
        self.assertNotIn("A:c1", state.aggregation_payload()["supported_claims"])
        self.assertIn("A:c1", state.aggregation_payload()["uncertain_claims"])

    def test_structured_genre_projection_is_exact_and_rejects_extra_facts(self):
        item = claim("The catalogue lists this film as Drama.")
        def judge(quote):
            return lambda _: {"verdict": "supported", "reason": "Bounded genre evidence.",
                "evidence_spans": [{"source_id": "movie:1", "quote": quote}], "usage": {}}
        self.assertEqual(EvidenceVerifier(self.sources, judge("Drama")).verify(item)["verdict"], "supported")
        with self.assertRaises(ValueError):
            EvidenceVerifier(self.sources, judge("Drama, Comedy")).verify(item)


class DurableProviderTests(unittest.TestCase):
    def test_provider_envelope_normalization_does_not_repair_business_fields(self):
        raw = {"status_code": 200, "elapsed_ms": 1, "body": json.dumps({
            "model": "fixture", "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "type": "json_object", "verdict": "supported", "reason": "x", "evidence_spans": []})}}]})}
        parsed = parse_raw(raw, {})
        self.assertNotIn("type", parsed["report"])
        self.assertEqual(parsed["report"]["evidence_spans"], [])
        self.assertEqual(parsed["report"]["verdict"], "supported")

    def test_fresh_and_resumed_chain_requests_are_identical(self):
        calls = []
        def transport(request):
            calls.append(request)
            return {"ranking": ["1"], "claims": [{"z": 1, "a": 2}], "updates": []}
        with tempfile.TemporaryDirectory() as directory:
            provider = DurableProvider(directory, {"version": "v5"}, transport, lambda raw, _: raw, True)
            first = provider.complete("initial", {"messages": [{"role": "user", "content": "x"}]})
            resumed = provider.complete("initial", {"messages": [{"role": "user", "content": "x"}]})
            # Includes JSON text embedded within the next request, the v4 failure.
            self.assertEqual(json.dumps(first), json.dumps(resumed))
            self.assertEqual(fingerprint({"content": canonical(first)}), fingerprint({"content": canonical(resumed)}))
            self.assertEqual(len(calls), 1)

    def test_paid_response_survives_parser_failure_and_replays_without_network(self):
        calls = []
        def transport(request):
            calls.append(request)
            return {"body": "bad JSON", "usage": {"prompt_tokens": 4}}
        def fail(raw, request):
            raise ValueError("schema error")
        with tempfile.TemporaryDirectory() as directory:
            provider = DurableProvider(directory, {"version": "v5"}, transport, fail, True)
            with self.assertRaises(ValueError):
                provider.complete("slot", {"prompt": "x"})
            self.assertTrue((Path(directory) / "slot.raw.json").exists())
            self.assertFalse((Path(directory) / "slot.intent.json").exists())
            repaired = DurableProvider(directory, {"version": "v5"}, transport,
                                       lambda raw, _: raw, False).complete("slot", {"prompt": "x"})
            self.assertEqual(repaired["body"], "bad JSON")
            self.assertEqual(len(calls), 1)

    def test_unknown_intent_does_not_retry(self):
        calls = []
        def fail(request):
            calls.append(request)
            raise RuntimeError("uncertain network failure")
        with tempfile.TemporaryDirectory() as directory:
            provider = DurableProvider(directory, {}, fail, lambda raw, _: raw, True)
            with self.assertRaises(RuntimeError):
                provider.complete("slot", {"prompt": "x"})
            with self.assertRaisesRegex(RuntimeError, "pending"):
                provider.complete("slot", {"prompt": "x"})
            self.assertEqual(len(calls), 1)

    def test_concurrent_collectors_cannot_duplicate_a_paid_slot(self):
        started, release = threading.Event(), threading.Event()
        calls = []
        def transport(request):
            calls.append(request)
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError("test collector release timed out")
            return {"ok": True}
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=2) as executor:
            first = DurableProvider(directory, {}, transport, lambda raw, _: raw, True)
            second = DurableProvider(directory, {}, transport, lambda raw, _: raw, True)
            future = executor.submit(first.complete, "slot", {"prompt": "x"})
            try:
                self.assertTrue(started.wait(timeout=3))
                with self.assertRaises(RuntimeError):
                    second.complete("slot", {"prompt": "x"})
            finally:
                release.set()
            self.assertEqual(future.result(timeout=3), {"ok": True})
            self.assertEqual(len(calls), 1)

    def test_missing_credentials_fail_before_intent_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = DurableProvider(directory, {}, RawDeepSeek(api_key=""), parse_raw, True)
            with self.assertRaisesRegex(RuntimeError, "API key"):
                provider.complete("slot", RequestsV5.envelope("ranking", {}, "system"))
            self.assertFalse(list(Path(directory).glob("*.intent.json")))

    def test_real_repair_entry_prepares_without_calls_and_refuses_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_v5("engineering", Path(directory))
            self.assertEqual(result["new_api_calls"], 0)
            self.assertEqual(result["cases"], 12)
            with self.assertRaisesRegex(ValueError, "calibration"):
                prepare_v5("pilot", Path(directory))

    def test_budget_is_enforced_before_provider_dispatch(self):
        class Never:
            def complete(self, *args):
                raise AssertionError("must not dispatch")
        client = BoundedClient(Never(), {"per_call_prompt_tokens": 100,
            "per_call_completion_tokens": 10, "total_prompt_tokens": 100, "total_completion_tokens": 10})
        with self.assertRaisesRegex(ValueError, "before dispatch"):
            client.complete("x", RequestsV5.envelope("patch", {"x": "large"}, "system", 10))

    def test_full_pipeline_initial_to_patch_to_verifier_to_final_survives_restart(self):
        owners = ["current_request", "history_applicability", "movie_fit"]
        evidence = {"case_id": "fixture:1", "observations": [{"source_id": "message:1", "text": "A request."}],
                    "previous_mentions_excluded_not_asserted_seen": [],
                    "candidates_in_shared_retrieval_order": [
                        {"source_id": "movie:" + str(i), "movie_id": str(i), "title": "Movie " + str(i),
                         "year": 2000, "genres": ["Drama"]} for i in range(1, 7)]}
        observed = []
        def transport(request):
            observed.append(request["kind"])
            payload = json.loads(request["messages"][1]["content"])
            if request["kind"] == "initial":
                report = {"ranking": ["1", "2", "3", "4", "5"], "claims": [claim()], "updates": []}
            elif request["kind"] == "patch":
                report = {"target_claim_id": payload["target_claim_id"],
                          "expected_hash": payload["expected_hash"], "replacement": None}
            else:
                self.assertEqual(set(payload["claim_state"]["retracted_ids"]),
                                 {owner + ":c1" for owner in owners})
                self.assertNotIn("2001", canonical(payload))
                report = {"ranking": ["2", "3", "4", "5", "6"]}
            return {"report": report, "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                    "tool_calls": 0, "tool_units": 0, "actual_cost": None, "retries": 0},
                    "model": "fixture", "elapsed_ms": 1}
        limits = {"per_call_prompt_tokens": 100000, "per_call_completion_tokens": 2000,
                  "total_prompt_tokens": 1000000, "total_completion_tokens": 100000}
        with tempfile.TemporaryDirectory() as directory:
            provider = DurableProvider(directory, {"version": "v5"}, transport, lambda raw, _: raw, True)
            first = run_case(evidence, BoundedClient(provider, limits))
            resumed = run_case(evidence, BoundedClient(provider, limits))
        self.assertEqual(len(observed), 7)
        self.assertEqual(first["final"], resumed["final"])
        self.assertEqual(first["claim_state"], resumed["claim_state"])
        self.assertFalse(first["initial_generation_imported"])

    def test_real_288_fact_fixtures_reach_actual_state_transitions(self):
        root = Path(__file__).parents[2] / "experiments/results/paired_deliberation_preflight"
        result = controlled_acceptance(root / "blind_annotation_packet_dev12.json",
                                       root / "automatic_evaluation_packet_dev12.json")
        self.assertTrue(result["pass"])
        self.assertEqual(result["cases"], 288)
        self.assertEqual(result["counts"], {"supported:keep": 144, "unsupported:retract": 144})
        self.assertFalse(result["evaluation_labels_available_to_method"])


if __name__ == "__main__":
    unittest.main()
