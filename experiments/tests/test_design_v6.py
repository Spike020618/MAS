import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.budget_v6 import AttemptClient, BudgetExceeded
from experiments.paired_deliberation.contracts_v6 import (OWNERS, METHODS, check_report, check_patch,
    initial_request, request_payload, review_request)
from experiments.paired_deliberation.coordination_v6 import Coordinator
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.client_v5 import parse_raw
from experiments.paired_deliberation.runner_v6 import run_method, run_suite, shared_initial
from experiments.paired_deliberation.state_v6 import StateV6
from experiments.paired_deliberation.verified_state import EvidenceVerifier


def evidence():
    return {"case_id": "fixture:6", "observations": [
        {"source_id": "message:1", "role": "seeker", "text": "Please recommend a movie."}],
        "previous_mentions_excluded_not_asserted_seen": [],
        "candidates_in_shared_retrieval_order": [{"source_id": "movie:" + str(i), "movie_id": str(i),
            "title": "Film " + str(i), "year": 2000, "genres": ["Drama"]} for i in range(1, 7)]}


def claim(cid="c1", text="Movie 1 has catalog year 2001.", kind="item_fact", deps=None, stance="support"):
    return {"claim_id": cid, "text": text, "source_ids": ["movie:1"], "candidate_ids": ["1"],
            "stance": stance, "claim_type": kind, "depends_on": deps or []}


def report(claims=None):
    return {"ranking": [str(i) for i in range(1, 6)], "claims": claims if claims is not None else [claim()], "updates": []}


LIMITS = {"max_calls": 20, "per_call_prompt_tokens": 64000, "per_call_completion_tokens": 1280,
          "total_prompt_tokens": 640000, "total_completion_tokens": 25600}


class Provider:
    def __init__(self):
        self.calls = []

    def complete(self, slot, request):
        self.calls.append((slot, copy.deepcopy(request)))
        p = request_payload(request)
        if request["kind"] == "initial_v6":
            c = claim(text="Movie 1 has catalog year 2000." if p["owner"] == "history_applicability"
                      else "Movie 1 has catalog year 2001.",
                      stance="oppose" if p["owner"] == "movie_fit" else "support")
            obj = report([c])
        elif request["kind"] == "patch_v6":
            obj = {k: p["schema"][k] for k in ("target_claim_id", "expected_hash", "replacement")}
        elif request["kind"] == "whole_review_v6":
            obj = report([copy.deepcopy(p["target"]["claim"])])
        elif request["kind"] == "select_v6":
            obj = {"target_claim_id": p["allowed_targets"][-1]}
        elif request["kind"] == "ranking_v6":
            obj = {"ranking": [str(i) for i in range(1, 6)]}
        elif request["kind"] == "verification":
            obj = {"verdict": "insufficient", "reason": "No decisive evidence.", "evidence_spans": []}
        else:
            raise AssertionError(request["kind"])
        return {"report": obj, "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": "offline-fixture", "elapsed_ms": 1}


class DesignV6Tests(unittest.TestCase):
    def setUp(self):
        self.e = evidence()

    def patch(self, state, target="current_request:c1", replacement=None):
        return {"target_claim_id": target, "expected_hash": fingerprint(state.claims[target]), "replacement": replacement}

    def test_refutation_invalidates_transitive_dependencies_without_deleting_other_claims(self):
        reports = {"current_request": report(), "history_applicability": report([
            claim("child", "Movie 1 has catalog year 2000.", deps=["current_request:c1"]),
            claim("other", "Movie 1 has catalog year 2000.")]), "movie_fit": report([
            claim("grandchild", "Movie 1 has catalog year 2000.", deps=["history_applicability:child"])])}
        state = StateV6(reports, self.e)
        v = EvidenceVerifier({"movie:1": self.e["candidates_in_shared_retrieval_order"][0]})
        state.apply("movie_fit", "history_applicability:other", self.patch(state, "history_applicability:other"), v)
        unchanged = copy.deepcopy(state.claims["history_applicability:other"])
        state.apply("movie_fit", "current_request:c1", self.patch(state), v)
        self.assertEqual(state.payload()["retracted_ids"], ["current_request:c1"])
        self.assertEqual(state.claims["history_applicability:child"]["status"], "invalidated")
        self.assertEqual(state.claims["movie_fit:grandchild"]["status"], "invalidated")
        self.assertEqual(state.claims["history_applicability:other"], unchanged)

    def test_replacement_is_atomic_scoped_and_stale_safe(self):
        s = StateV6({"current_request": report()}, self.e)
        v = EvidenceVerifier({"movie:1": self.e["candidates_in_shared_retrieval_order"][0]})
        initial = copy.deepcopy(s.claims)
        with self.assertRaisesRegex(ValueError, "new support"):
            s.apply("current_request", "current_request:c1", self.patch(s, replacement=claim(text="Movie 1 has catalog year 1999.")), v)
        self.assertEqual(s.claims, initial)
        bad = claim(text="Movie 1 has catalog year 2000.", kind="explicit_preference")
        with self.assertRaisesRegex(ValueError, "scope"):
            s.apply("current_request", "current_request:c1", self.patch(s, replacement=bad), v)
        patch = self.patch(s, replacement=claim(text="Movie 1 has catalog year 2000."))
        s.apply("current_request", "current_request:c1", patch, v)
        self.assertEqual(s.claims["current_request:c1"]["status"], "supported")
        with self.assertRaisesRegex(ValueError, "hash"):
            s.apply("current_request", "current_request:c1", patch, v)

    def test_hypothesis_is_never_certified_or_sent_to_verifier(self):
        s = StateV6({"current_request": report([claim(kind="preference_hypothesis")])}, self.e)
        class Fail:
            def verify(self, c):
                raise AssertionError("must not verify hypothetical preference truth")
        s.apply("current_request", "current_request:c1", self.patch(s), Fail())
        self.assertFalse(s.payload()["supported_claims"])
        self.assertEqual(s.claims["current_request:c1"]["status"], "hypothesis")

    def test_cycles_and_missing_dependencies_rejected(self):
        for dep in ["current_request:c1", "other:missing"]:
            with self.assertRaises(ValueError):
                StateV6({"current_request": report([claim(deps=[dep])])}, self.e)

    def test_local_dependencies_bind_only_to_current_owner_and_keep_raw_report(self):
        reports = {"current_request": report([claim(), claim("c2", deps=["c1"])]),
                   "movie_fit": report([claim()])}
        original = copy.deepcopy(reports)
        state = StateV6(reports, self.e)
        self.assertEqual(state.claims["current_request:c2"]["claim"]["depends_on"], ["current_request:c1"])
        self.assertEqual(reports, original)
        with self.assertRaisesRegex(ValueError, "unknown claim dependency"):
            StateV6({"current_request": report([claim(deps=["missing"])])}, self.e)
        with self.assertRaisesRegex(ValueError, "duplicate resolved"):
            StateV6({"current_request": report([claim(), claim("c2", deps=["c1", "current_request:c1"])])}, self.e)

    def test_real_initial_failure_replays_without_rewriting_raw_response(self):
        path = Path("experiments/results/paired_deliberation_d1_v6_engineering_r1/slots/redial_5431_shared_initial_current_request.json")
        if not path.exists():
            self.skipTest("live engineering fixture unavailable")
        original = json.loads(path.read_text())["record"]["report"]
        # Its candidate pool is the saved public input, not fixture candidates.
        e = json.loads((path.parents[1] / "inputs/redial_5431.json").read_text())
        before = copy.deepcopy(original)
        check_report(original, e)
        state = StateV6({"current_request": original}, e)
        self.assertEqual(state.claims["current_request:c3"]["claim"]["depends_on"], ["current_request:c1"])
        self.assertEqual(original, before)

    def test_whole_report_is_unconstrained_but_removed_premise_invalidates_dependents(self):
        s = StateV6({"current_request": report(), "movie_fit": report([
            claim(deps=["current_request:c1"])])}, self.e)
        s.replace_owner("current_request", report([claim("new", "Movie 1 has catalog year 2000.")]))
        self.assertNotIn("current_request:c1", s.claims)
        self.assertEqual(s.claims["movie_fit:c1"]["status"], "invalidated")
        self.assertEqual(s.claims["current_request:new"]["status"], "unverified")

    def test_fixed_and_adaptive_choose_different_observable_targets(self):
        reports = {owner: report([claim(stance="unknown" if owner == "movie_fit" else "support")]) for owner in OWNERS}
        s = StateV6(reports, self.e)
        f, a = Coordinator("fixed", reports, 1), Coordinator("adaptive", reports, 1)
        self.assertEqual(f.next(s), "current_request:c1")
        self.assertEqual(a.next(s), "movie_fit:c1")
        self.assertIsNone(a.next(s))

    def test_all_five_conditions_run_with_actual_state_changes_and_on_demand_checks(self):
        provider = Provider()
        shared = shared_initial(self.e, AttemptClient(provider, LIMITS), "shared")
        self.assertTrue(all(request["kind"] == "initial_v6" for _, request in provider.calls))
        result = run_suite(self.e, lambda _: AttemptClient(provider, LIMITS), shared)
        self.assertEqual(set(result), set(METHODS))
        self.assertTrue(all(row["status"] == "completed" for row in result.values()))
        full = result["verified_fixed"]["result"]
        self.assertIn("current_request:c1", full["state"]["retracted_ids"])
        self.assertIn("history_applicability:c1", full["state"]["supported_claims"])
        self.assertEqual(result["peer"]["result"]["state"]["retracted_ids"], [])
        self.assertEqual(result["central"]["result"]["schedule"][0]["target"], "movie_fit:c1")
        self.assertEqual(full["usage"]["attempts"], 7)  # 3 shared + 3 reviews + aggregate; bounded facts free.
        self.assertEqual(len(full["verification_cache"]), 3)  # Two texts plus differing stance identity.

    def test_central_end_to_end_does_not_generate_three_role_reports(self):
        provider = Provider()
        result = run_method(self.e, "central", AttemptClient(provider, LIMITS), "central", max_rounds=1)
        initial = [request_payload(req)["owner"] for _, req in provider.calls if req["kind"] == "initial_v6"]
        self.assertEqual(initial, ["central"])
        self.assertFalse(result["shared_initial"])

    def test_budget_accounts_failed_attempt_and_unknown_usage(self):
        class Failing:
            def complete(self, slot, request):
                raise RuntimeError("transport interrupted; billing unknown")
        client = AttemptClient(Failing(), {**LIMITS, "max_calls": 1})
        with self.assertRaises(RuntimeError):
            client.complete("one", initial_request(self.e, OWNERS[0]))
        self.assertFalse(client.summary()["usage_complete"])
        self.assertIsNone(client.summary()["prompt_tokens"]["total"])
        with self.assertRaises(BudgetExceeded):
            client.complete("two", initial_request(self.e, OWNERS[0]))
        self.assertEqual(client.summary()["attempts"], 1)

    def test_budget_stops_before_review_and_keeps_final_slot(self):
        provider = Provider()
        limits = {**LIMITS, "max_calls": 4}
        result = run_method(self.e, "verified_fixed", AttemptClient(provider, limits), "tight")
        self.assertEqual(result["stop"], "budget_reserved_for_final")
        self.assertEqual(result["usage"]["attempts"], 4)
        self.assertFalse(result["schedule"])

    def test_historical_contract_failures_stay_invalid_and_prompts_are_explicit(self):
        too_many = report([claim(str(i)) for i in range(4)])
        with self.assertRaisesRegex(ValueError, "three"):
            check_report(too_many, self.e)
        s = StateV6({"current_request": report()}, self.e)
        p = self.patch(s)
        for invalid in [{"schema": p}, {k: v for k, v in p.items() if k != "expected_hash"}]:
            with self.assertRaises(ValueError):
                check_patch(invalid, "current_request:c1", fingerprint(s.claims["current_request:c1"]))
        request = review_request(self.e, s, "current_request:c1", "current_request")
        self.assertIn("expected_hash is mandatory", request["messages"][0]["content"])
        self.assertIn("at most three claims", initial_request(self.e, OWNERS[0])["messages"][0]["content"])

    def test_label_root_and_shared_tampering_fail_before_dispatch(self):
        provider = Provider()
        with self.assertRaisesRegex(ValueError, "evidence root"):
            run_method({**self.e, "gold": "1"}, "independent", AttemptClient(provider, LIMITS), "bad")
        self.assertEqual(provider.calls, [])
        shared = shared_initial(self.e, AttemptClient(provider, LIMITS), "shared")
        shared["initial"][OWNERS[0]]["claims"][0]["text"] = "tampered"
        with self.assertRaisesRegex(ValueError, "integrity"):
            run_method(self.e, "peer", AttemptClient(provider, LIMITS), "bad", shared)

    def test_raw_first_fresh_and_resume_equivalent_in_new_client(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def transport(req):
                calls.append(req)
                return {"status_code": 200, "elapsed_ms": 1, "body": json.dumps({"model": "fixture", "usage": {
                    "prompt_tokens": 5, "completion_tokens": 2}, "choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(report())}}]})}
            p = DurableProvider(directory, {"version": "v6-test"}, transport, parse_raw, True)
            req = initial_request(self.e, OWNERS[0])
            first = AttemptClient(p, LIMITS).complete("slot", req)
            second = AttemptClient(p, LIMITS).complete("slot", req)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            self.assertTrue((Path(directory) / "slot.raw.json").exists())

    def test_semantic_refusal_is_not_a_contract_failure_or_successful_replacement(self):
        from experiments.paired_deliberation.state_v6 import ReplacementPreconditionError
        s = StateV6({"current_request": report([claim(text="Movie 1 has catalog year 2000.")])}, self.e)
        old = copy.deepcopy(s.claims)
        v = EvidenceVerifier({"movie:1": self.e["candidates_in_shared_retrieval_order"][0]})
        with self.assertRaises(ReplacementPreconditionError):
            s.apply("current_request", "current_request:c1", self.patch(s, replacement=claim(text="Movie 1 has catalog year 2000.")), v)
        self.assertEqual(s.claims, old)
        self.assertEqual(s.events, [])


if __name__ == "__main__":
    unittest.main()
