import json
import tempfile
import unittest
from pathlib import Path

from experiments.paired_deliberation.ledger import ExperimentLedger
from experiments.paired_deliberation.runner import PairedRunner
from experiments.paired_deliberation.audit_existing import audit
from experiments.paired_deliberation.redial_adapter import (
    AnnotatedProcessEvaluator,
    EvidenceContractChecker,
)
from experiments.paired_deliberation.prepare_annotations import prepare
from experiments.paired_deliberation.automatic_evaluation import (
    ProxyJudgePanel,
    deterministic_contract_audit,
    synthetic_fact_cases,
)
from experiments.paired_deliberation.prepare_automatic_packet import prepare as prepare_automatic
from experiments.paired_deliberation.pipeline import run_e1_method, run_e2_arm
from experiments.paired_deliberation.plan import build_plan
from experiments.paired_deliberation.provider import CheckpointedProvider
from experiments.paired_deliberation.acceptance import run_acceptance
from experiments.paired_deliberation.deepseek_transport import DeepSeekTransport
from experiments.paired_deliberation.redial_requests import RedialRequestBuilder
from experiments.paired_deliberation.run_experiment import run as prepare_or_run_experiment


class PairedDeliberationTests(unittest.TestCase):
    def setUp(self):
        self.initial = {"A": {"choice": "x"}, "B": {"choice": "y"}}
        self.obs = {"A": {"record": "a"}, "B": {"record": "b"}}
        self.requests = []
        self.budget = {
            "per_call_prompt_tokens": 20,
            "per_call_completion_tokens": 10,
            "total_prompt_tokens": 40,
            "total_completion_tokens": 20,
            "total_tool_calls": 2,
        }

        def review(request):
            self.requests.append(request)
            return {
                "report": {"choice": "z"},
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "tool_calls": 1},
            }

        def unresolved(*_args):
            return {
                "decision": "unresolved",
                "target_claim_ids": [],
                "evidence_refs": [],
                "rationale": "available evidence does not decide the claim",
            }

        self.runner = PairedRunner(
            review,
            lambda state: [state[key]["choice"] for key in sorted(state)],
            unresolved,
        )

    def test_same_initial_and_complete_budget_contract(self):
        results = [
            self.runner.run(self.initial, self.obs, arm, ["A", "B"], self.budget)
            for arm in ("self_review", "peer_review", "evidence_update")
        ]
        self.assertEqual(len({result["initial_hash"] for result in results}), 1)
        self.assertTrue(all(result["review_calls"] == 2 for result in results))
        self.assertTrue(all(result["usage"]["prompt_tokens"] == 24 for result in results))
        self.assertEqual(self.initial, {"A": {"choice": "x"}, "B": {"choice": "y"}})

    def test_frozen_peer_reviews_do_not_observe_prior_update(self):
        result = self.runner.run(
            self.initial, self.obs, "peer_review", ["A", "B"], self.budget, "frozen"
        )
        self.assertEqual(result["events"][0]["visible_report_hash"], result["events"][1]["visible_report_hash"])

    def test_sequential_peer_review_observes_prior_update(self):
        result = self.runner.run(
            self.initial, self.obs, "peer_review", ["A", "B"], self.budget, "sequential"
        )
        self.assertNotEqual(result["events"][0]["visible_report_hash"], result["events"][1]["visible_report_hash"])

    def test_self_review_has_no_peer_report(self):
        self.runner.run(self.initial, self.obs, "self_review", ["A"], self.budget)
        self.assertEqual(set(self.requests[0]["reports"]), {"A"})

    def test_unresolved_update_is_audited_and_does_not_overwrite(self):
        result = self.runner.run(self.initial, self.obs, "evidence_update", ["A"], self.budget)
        self.assertEqual(result["final_reports"], self.initial)
        self.assertEqual(result["events"][0]["check"]["decision"], "unresolved")
        self.assertTrue(result["events"][0]["check"]["rationale"])

    def test_accepted_update_requires_target_and_evidence(self):
        runner = PairedRunner(
            self.runner.review,
            lambda state: state,
            lambda *_: {
                "decision": "accept",
                "target_claim_ids": [],
                "evidence_refs": [],
                "rationale": "unsupported acceptance",
            },
        )
        with self.assertRaises(ValueError):
            runner.run(self.initial, self.obs, "evidence_update", ["A"], self.budget)

    def test_no_checker_cannot_pretend_evidence_protocol_complete(self):
        runner = PairedRunner(lambda _: {}, lambda state: state)
        with self.assertRaises(NotImplementedError):
            runner.run(self.initial, self.obs, "evidence_update", ["A"], self.budget)

    def test_independent_arm_uses_no_review_and_no_scheduled_cap(self):
        result = self.runner.run(self.initial, self.obs, "independent", ["A", "B"], self.budget)
        self.assertEqual(result["review_calls"], 0)
        self.assertEqual(result["scheduled_review_calls"], 0)
        self.assertEqual(self.requests, [])

    def test_missing_usage_is_not_free(self):
        runner = PairedRunner(lambda _: {"report": {}}, lambda state: state)
        with self.assertRaises(ValueError):
            runner.run(self.initial, self.obs, "peer_review", ["A"], self.budget)

    def test_each_budget_dimension_is_enforced(self):
        for key, value in (
            ("prompt_tokens", 21),
            ("completion_tokens", 11),
            ("tool_calls", 3),
        ):
            usage = {"prompt_tokens": 1, "completion_tokens": 1, "tool_calls": 0}
            usage[key] = value
            runner = PairedRunner(
                lambda _, usage=usage: {"report": {}, "usage": usage},
                lambda state: state,
            )
            with self.assertRaises(ValueError):
                runner.run(self.initial, self.obs, "peer_review", ["A"], self.budget)

    def test_frozen_mode_rejects_duplicate_agent_schedule(self):
        with self.assertRaises(ValueError):
            self.runner.run(self.initial, self.obs, "peer_review", ["A", "A"], self.budget)

    def test_preflight_protocol_has_strong_baselines_and_publication_policy(self):
        path = Path(__file__).parents[1] / "paired_deliberation" / "protocol.json"
        protocol = json.loads(path.read_text())
        self.assertIn("same_agent_independent_samples_shared_aggregator", protocol["e1_information_baselines"])
        self.assertIn("centralized_agent_all_evidence_and_tool_outputs", protocol["e1_information_baselines"])
        self.assertEqual(len(protocol["e3_ablations"]), 5)
        self.assertFalse(protocol["paid_or_reserved_test_authorized"])
        self.assertGreaterEqual(
            len(protocol["published_result_policy"]["direct_numeric_comparison_requires"]), 5
        )

    def test_experiment_ledger_requires_complete_path_and_sums_every_dimension(self):
        ledger = ExperimentLedger()
        for stage in ("initial_generation", "review", "aggregation"):
            ledger.add(stage, 10, 2, 1, 50, 0, 100, "0.001", {"arm": "peer"})
        total = ledger.assert_complete_path()
        self.assertEqual(total["prompt_tokens"], 30)
        self.assertEqual(total["tool_units"], 150)
        self.assertEqual(total["actual_cost"], "0.003")

    def test_experiment_ledger_does_not_accept_missing_cost_or_metadata(self):
        ledger = ExperimentLedger()
        ledger.add("review", 1, 1, 0, 0, 0, 1, None, {})
        self.assertIsNone(ledger.summary()["actual_cost"])
        self.assertFalse(ledger.summary()["actual_cost_complete"])
        with self.assertRaises(ValueError):
            ledger.add("review", 1, 1, 0, 0, 0, 1, "0", None)

    def test_historical_pilot_is_not_mislabeled_as_new_protocol_replay(self):
        report = audit("pilot")
        self.assertFalse(report["compatible_as_new_e2_effect_replay"])
        self.assertEqual(report["counts"]["cases"], 48)
        self.assertGreater(report["counts"]["cases_with_duplicate_target_role"], 0)
        self.assertFalse(report["reserved_labels_read"])
        self.assertEqual(report["new_model_calls"], 0)

    def test_execution_checker_enforces_contract_without_semantic_oracle(self):
        checker = EvidenceContractChecker("redial:fixture", {"message:1"})
        proposal = {
            "updates": [{
                "target_claim_id": "A:c1",
                "decision": "accept",
                "reason": "supported",
                "source_ids": ["message:1"],
            }]
        }
        result = checker("A", {}, proposal, {}, {})
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["target_claim_ids"], ["A:c1"])

    def test_blind_annotation_is_used_only_after_execution(self):
        evaluator = AnnotatedProcessEvaluator(
            {"message:1"},
            {
                "A:c1": {
                    "verdict": "unsupported",
                    "source_ids": ["message:1"],
                    "annotator_ids": ["blind-1", "blind-2"],
                    "adjudication": "The claim exceeds the message.",
                }
            },
        )
        result = evaluator.evaluate([{
            "check": {"target_claim_ids": ["A:c1"]},
            "decision": "accept",
        }])
        self.assertEqual(result["counts"]["incorrect_claim_preserved"], 1)
        self.assertFalse(result["annotations_used_during_execution"])

    def test_annotation_packet_hides_method_outputs_and_evaluation_labels(self):
        packet = prepare("pilot", 2)
        self.assertEqual(len(packet["cases"]), 2)
        self.assertFalse(packet["reserved_labels_read"])
        self.assertEqual(packet["new_model_calls"], 0)
        serialized = json.dumps(packet["cases"])
        self.assertNotIn('"R5"', serialized)
        self.assertNotIn('"ranking"', serialized)
        self.assertNotIn("relevant_movie_ids", serialized)
        self.assertTrue(all(claim["annotation"] is None
                            for case in packet["cases"] for claim in case["claims"]))

    def test_deterministic_audit_does_not_claim_semantic_support(self):
        result = deterministic_contract_audit(
            {"text": "claim", "source_ids": ["message:1"], "candidate_ids": ["m1"]},
            {"message:1"}, {"m1"},
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["semantic_support"], "not_evaluated")

    def test_synthetic_fact_cases_have_programmatic_positive_and_negative_labels(self):
        result = synthetic_fact_cases([
            {"source_id": "movie:m1", "movie_id": "m1", "year": 2000, "genres": ["Drama"]},
            {"source_id": "movie:m2", "movie_id": "m2", "year": 2001, "genres": ["Comedy"]},
        ])
        self.assertEqual({row["verdict"] for row in result["cases"]}, {"supported", "unsupported"})
        self.assertIn("bounded", result["validity"])

    def test_proxy_panel_is_blind_and_uses_third_judge_only_on_disagreement(self):
        def judge(verdict):
            return lambda _: {"verdict": verdict, "reason": "proxy reason", "usage": {"tokens": 1}}
        panel = ProxyJudgePanel([("j1", judge("supported")), ("j2", judge("unsupported")),
                                 ("j3", judge("unclear"))])
        result = panel.evaluate({"claim_text": "x", "evidence": "y"})
        self.assertTrue(result["third_judge_used"])
        self.assertEqual(result["verdict"], "unclear")
        self.assertEqual(result["label_type"], "model_proxy_not_human_gold")
        with self.assertRaises(ValueError):
            panel.evaluate({"claim_text": "x", "arm": "proposed"})

    def test_automatic_packet_makes_no_human_gold_claim(self):
        root = Path(__file__).parents[2]
        packet = prepare_automatic(
            root / "experiments/results/paired_deliberation_preflight/blind_annotation_packet_dev12.json"
        )
        self.assertFalse(packet["human_gold_claimed"])
        self.assertFalse(packet["reserved_labels_read"])
        self.assertEqual(packet["new_model_calls"], 0)
        self.assertEqual(len(packet["deterministic_contract_results"]), 69)
        self.assertGreater(sum(len(case["cases"]) for case in packet["controlled_corruption_cases"]), 0)

    def test_execution_plan_freezes_all_required_baselines_and_ablations(self):
        plan = build_plan(["case:1"], "fixed-model", self.budget, review_rounds=2)
        self.assertIn("E1_same_agent_samples", plan["methods"])
        self.assertIn("E1_central_all_evidence", plan["methods"])
        self.assertIn("E2_evidence_update", plan["methods"])
        self.assertEqual(len(plan["ablations"]), 5)
        self.assertFalse(plan["labels_available_to_execution"])
        self.assertFalse(plan["provider_calls_authorized"])

    def test_checkpoint_provider_is_inert_then_reuses_exact_checkpoint(self):
        calls = []
        def transport(request):
            calls.append(request)
            return {"report": {"choice": "z"},
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "tool_calls": 0,
                              "tool_units": 0, "actual_cost": "0.001", "retries": 0},
                    "model": "fixed-model", "elapsed_ms": 3}
        with tempfile.TemporaryDirectory() as directory:
            blocked = CheckpointedProvider(directory, "plan", transport, authorized=False)
            with self.assertRaises(RuntimeError):
                blocked.complete("slot", {"prompt": "x"})
            provider = CheckpointedProvider(directory, "plan", transport, authorized=True)
            first = provider.complete("slot", {"prompt": "x"})
            second = provider.complete("slot", {"prompt": "x"})
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            with self.assertRaises(ValueError):
                provider.complete("slot", {"prompt": "changed"})

    def test_e2_pipeline_accounts_shared_initial_review_and_aggregation(self):
        def transport(request):
            if request.get("kind") == "aggregate":
                report = {"ranking": ["m1"]}
            else:
                report = {"choice": "z"}
            return {"report": report,
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "tool_calls": 0,
                              "tool_units": 0, "actual_cost": "0.001", "retries": 0},
                    "model": "fixed-model", "elapsed_ms": 3}
        initial_record = lambda choice: {
            "report": {"choice": choice},
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "tool_calls": 0,
                      "tool_units": 0, "actual_cost": "0.001", "retries": 0},
            "model": "fixed-model", "elapsed_ms": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            provider = CheckpointedProvider(directory, "plan", transport, authorized=True)
            result = run_e2_arm(
                "case:1", {"A": initial_record("x"), "B": initial_record("y")}, self.obs,
                "peer_review", ["A", "B"], self.budget, "frozen", provider,
                lambda _case, request: {"kind": "review", "request": request},
                lambda _case, _arm, state: {"kind": "aggregate", "state": state},
            )
        self.assertEqual(result["ledger"]["stages"], ["aggregation", "initial_generation", "review"])
        self.assertEqual(result["ledger"]["records"], 5)
        self.assertEqual(result["model"], "fixed-model")
        self.assertFalse(result["evaluation_labels_available_to_execution"])

    def test_frozen_and_sequential_paths_do_not_share_checkpoint_slots(self):
        def transport(request):
            report = ({"ranking": ["m1"]} if request.get("kind") == "aggregate"
                      else {"choice": "z"})
            return {"report": report,
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "tool_calls": 0,
                              "tool_units": 0, "actual_cost": "0", "retries": 0},
                    "model": "fixed-model", "elapsed_ms": 1}
        initial = {key: {"report": value,
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "tool_calls": 0,
                                   "tool_units": 0, "actual_cost": "0", "retries": 0},
                         "model": "fixed-model", "elapsed_ms": 1}
                   for key, value in self.initial.items()}
        with tempfile.TemporaryDirectory() as directory:
            provider = CheckpointedProvider(directory, "plan", transport, True)
            for visibility, path_name in (("frozen", "peer_review"),
                                          ("sequential", "peer_sequential")):
                run_e2_arm("case:1", initial, self.obs, "peer_review", ["A", "B"],
                           self.budget, visibility, provider,
                           lambda _case, request: {"kind": "review", "request": request},
                           lambda _case, _arm, state: {"kind": "aggregate", "state": state},
                           path_name=path_name)
            names = {path.name for path in Path(directory).glob("*.json")}
            self.assertTrue(any("peer_review" in name for name in names))
            self.assertTrue(any("peer_sequential" in name for name in names))

    def test_e1_pipeline_executes_all_strong_baseline_shapes(self):
        def transport(request):
            report = {"ranking": ["m1"]} if request.get("kind") == "aggregate" else {"choice": "x"}
            return {"report": report,
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "tool_calls": 0,
                              "tool_units": 0, "actual_cost": "0", "retries": 0},
                    "model": "fixed-model", "elapsed_ms": 1}
        with tempfile.TemporaryDirectory() as directory:
            for method in ("single_once", "single_serial", "same_agent_samples",
                           "multi_role", "central_all_evidence"):
                provider = CheckpointedProvider(Path(directory) / method, "plan", transport, True)
                result = run_e1_method(
                    "case:1", method, ["A", "B", "C"], 2, provider,
                    lambda _case, name, stage, identity, prior: {
                        "kind": "review", "method": name, "stage": stage,
                        "identity": identity, "prior": prior,
                    },
                    lambda _case, name, reports: {"kind": "aggregate", "method": name,
                                                   "reports": reports}, self.budget,
                )
                self.assertEqual(result["model"], "fixed-model")
                self.assertFalse(result["evaluation_labels_available_to_execution"])

    def test_full_offline_acceptance_covers_every_e2_arm(self):
        root = Path(__file__).parents[2]
        result = run_acceptance(
            root / "experiments/results/paired_deliberation_preflight/automatic_evaluation_packet_dev12.json"
        )
        self.assertTrue(result["pass"])
        self.assertEqual(len(result["e1_methods"]), 5)
        self.assertEqual(len(result["arms"]), 6)
        self.assertEqual(result["synthetic"]["cases"], 288)
        self.assertEqual(result["provider_calls"], 0)
        self.assertFalse(result["reserved_labels_read"])

    def test_redial_request_builder_keeps_labels_out_and_supports_all_stages(self):
        evidence = {"case_id": "case:1", "observations": [{"source_id": "message:1", "text": "x"}],
                    "previous_mentions_excluded_not_asserted_seen": [],
                    "candidates_in_shared_retrieval_order": [
                        {"source_id": "movie:m%d" % i, "movie_id": "m%d" % i,
                         "title": "T%d" % i, "year": 2000 + i, "genres": ["Drama"]}
                        for i in range(1, 6)]}
        builder = RedialRequestBuilder(evidence)
        initial = builder.e1("case:1", "single_once", "initial", "single_agent", None)
        self.assertFalse(initial["labels_available"])
        self.assertNotIn("gold", json.dumps(initial))
        reports = {"A": {"ranking": ["m1", "m2", "m3", "m4", "m5"],
                           "claims": [{"claim_id": "c1", "text": "x", "source_ids": ["message:1"],
                                      "candidate_ids": ["m1"], "stance": "support"}], "updates": []}}
        request = builder.e2_review("case:1", {"agent": "A", "mode": "peer_review",
                                                "reports": reports})
        aggregate = builder.aggregate("case:1", "peer_review", reports)
        sampled = builder.aggregate("case:1", "same_agent_samples", list(reports.values()))
        self.assertFalse(request["labels_available"])
        self.assertFalse(aggregate["labels_available"])
        self.assertFalse(sampled["labels_available"])

    def test_deepseek_transport_without_key_is_inert(self):
        transport = DeepSeekTransport(api_key="")
        with self.assertRaises(RuntimeError):
            transport({"labels_available": False})

    def test_real_experiment_entrypoint_prepares_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_or_run_experiment("engineering", Path(directory), False)
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(result["provider_calls"], 0)
            self.assertFalse(result["reserved_test_authorized"])
            plan = json.loads((Path(directory) / "plan.json").read_text())
            self.assertFalse(plan["provider_calls_authorized"])


if __name__ == "__main__":
    unittest.main()
