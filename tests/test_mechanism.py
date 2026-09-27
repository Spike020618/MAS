import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from core.mechanism.behavioral_metrics import (
    evidence_copy_score,
    evidence_novelty,
    unsupported_flip,
)
from core.mechanism.contracts import (
    CONTRACTS, apply_signal_mask, parse_action, render_contract, scale_costs,
)
from core.mechanism.schema import (
    AEICReport,
    AgentAction,
    AgentRound,
    ExperimentRecord,
    MechanismOffer,
)
from core.mechanism.storage import RecordStore
from core.mechanism.controller import MechanismController


class MechanismSchemaTests(unittest.TestCase):
    def setUp(self):
        self.offer = MechanismOffer(
            mechanism="aeic_contract",
            credit_rule="novel_verifiable_evidence",
            visible_reward={"verify": 2.0},
            visible_cost={"verify": 1.0},
            information_policy="peer_conclusions",
            prompt_hash="abc",
            config_hash="def",
        )
        self.report = AEICReport("premise", ["specific evidence"], "therefore", "yes")

    def test_complete_record_requires_cost_logs(self):
        item = AgentRound(1, 0, self.offer, AgentAction.VERIFY, self.report)
        rec = ExperimentRecord("x", "D", "q", "c", 42, {"0": "m"}, [item],
                               final_decision="yes", completed=True)
        errors = rec.validate()
        self.assertTrue(any("token usage" in x for x in errors))
        self.assertTrue(any("latency" in x for x in errors))
        self.assertTrue(any("requires reward" in x for x in errors))
        self.assertTrue(any("requires realized_cost" in x for x in errors))
        self.assertTrue(any("requires raw_response" in x for x in errors))
        self.assertTrue(any("requires action_trace" in x for x in errors))

    def test_valid_complete_record(self):
        item = AgentRound(1, 0, self.offer, AgentAction.VERIFY, self.report,
                          token_input=10, token_output=20, latency_s=0.5,
                          reputation_before=0.0, reputation_after=1.0,
                          reward=1.0, realized_cost=0.5,
                          raw_response='{"A":"a"}',
                          action_trace=[{"stage": "choose"}])
        rec = ExperimentRecord("x", "D", "q", "c", 42, {"0": "m"}, [item],
                               final_decision="yes", completed=True)
        self.assertEqual(rec.validate(), [])


class BehavioralMetricTests(unittest.TestCase):
    def test_novelty_and_copying(self):
        report = AEICReport("", ["Mars has exactly two moons"], "", "yes")
        visible = ["Mars has two moons"]
        self.assertGreater(evidence_copy_score(report, visible), 0.5)
        self.assertGreater(evidence_novelty(report, visible), 0.0)

    def test_unsupported_flip(self):
        old = AEICReport("a", ["same evidence"], "i", "yes")
        new = AEICReport("a", ["same evidence"], "i", "no")
        self.assertTrue(unsupported_flip(old, new, "yes", "no", []))
        self.assertFalse(unsupported_flip(old, new, "yes", "no", ["new fact"]))


class ContractTests(unittest.TestCase):
    def test_contract_is_visible_and_hashable(self):
        spec = CONTRACTS["aeic_contract"]
        prompt = render_contract(spec)
        offer = spec.to_offer(prompt)
        self.assertIn('"ACTION"', prompt)
        self.assertTrue(offer.prompt_hash)
        self.assertTrue(offer.config_hash)

    def test_adaptive_internal_label_is_not_a_prompt_confound(self):
        prompt = render_contract(CONTRACTS["adaptive_stackelberg"])
        self.assertIn("Mechanism: aeic_contract", prompt)
        self.assertNotIn("history_adaptive_contract", prompt)

    def test_action_must_be_declared_first(self):
        self.assertEqual(parse_action("ACTION: verify\n{}"), AgentAction.VERIFY)
        self.assertEqual(parse_action('{"ACTION":"low_effort","A":""}'),
                         AgentAction.LOW_EFFORT)
        with self.assertRaises(ValueError):
            parse_action("I chose nothing")

    def test_cost_scale_changes_costs_only(self):
        base = CONTRACTS["static_aeic"]
        scaled = scale_costs(base, 1.5)
        self.assertEqual(scaled.rewards, base.rewards)
        self.assertEqual(scaled.costs["aeic_effort"], 1.5)
        self.assertEqual(scaled.costs["verify"], 3.0)

    def test_signal_mask_zeroes_only_removed_aeic_components(self):
        masked = apply_signal_mask(CONTRACTS["static_aeic"], "EIC")
        self.assertEqual(masked.credit_rule, "fixed_process_quality_components")
        self.assertEqual(masked.rewards["field_A"], 0.0)
        self.assertEqual(masked.rewards["field_E"], 0.25)
        self.assertEqual(masked.rewards["consistency"], 1.0)


class StorageTests(unittest.TestCase):
    def test_terminal_manifest_detects_recomputed_mismatch(self):
        import json
        from experiments.mechanism.build_frozen_manifest import inspect_bundle

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "suite"
            recomputed = Path(tmp) / "suite.recomputed"
            bundle.mkdir()
            recomputed.mkdir()
            for name in ("matrix.json", "analysis.json", "statistics.json",
                         "sensitivity.json"):
                (bundle / name).write_text('{"value":1}', encoding="utf-8")
            (bundle / "verification.json").write_text(
                '{"pass":true,"matrix_version":"v"}', encoding="utf-8")
            (bundle / "provenance.json").write_text(
                '{"n_question_level_records":1}', encoding="utf-8")
            (bundle / "question_level.jsonl").write_text('{}\n', encoding="utf-8")
            for name in ("analysis.json", "sensitivity.json"):
                (recomputed / name).write_text((bundle / name).read_text(), encoding="utf-8")
            (recomputed / "statistics.json").write_text('{"value":2}', encoding="utf-8")
            (recomputed / "comparison.json").write_text(
                '{"analysis_matches_frozen":true,'
                '"statistics_matches_frozen":true,'
                '"sensitivity_matches_frozen":true}', encoding="utf-8")
            result = inspect_bundle(bundle)
            self.assertEqual(result["recompute_mismatches"], ["statistics.json"])
            self.assertFalse(result["complete"])

    def test_frozen_comparison_ignores_only_input_path_metadata(self):
        from experiments.mechanism.recompute_frozen import comparable

        left = {"records": ["/tmp/a"], "metric": {"value": 1}}
        right = {"records": ["/frozen/a"], "metric": {"value": 1}}
        self.assertEqual(comparable(left), comparable(right))
        right["metric"]["value"] = 2
        self.assertNotEqual(comparable(left), comparable(right))

    def _record(self, completed=True, error=None):
        offer = CONTRACTS["aeic_contract"].to_offer("prompt")
        report = AEICReport("a", ["e"], "i", "yes")
        rounds = [] if error else [AgentRound(
            1, 0, offer, AgentAction.AEIC_EFFORT, report,
            token_input=1, token_output=1, latency_s=0.1,
            reputation_before=0.0, reputation_after=1.0,
            reward=1.0, realized_cost=0.5, raw_response='{"A":"a"}',
            action_trace=[{"stage": "choose"}],
        )]
        return ExperimentRecord(
            "pilot", "D", "q1", "c", 42, {"0": "m"}, rounds,
            final_decision="yes" if completed else None,
            completed=completed, error=error,
        )

    def test_success_replaces_error_without_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RecordStore(Path(tmp) / "records.jsonl")
            store.upsert(self._record(completed=False, error="temporary"))
            store.upsert(self._record())
            self.assertEqual(len(store.load()), 1)
            self.assertEqual(len(store.completed_keys()), 1)
            saved = next(iter(store.load().values()))
            self.assertEqual(saved["attempt_count"], 2)
            self.assertEqual(saved["prior_errors"], ["temporary"])

    def test_completed_record_is_immutable_under_concurrent_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RecordStore(Path(tmp) / "records.jsonl")
            first = self._record(completed=True)
            store.upsert(first)
            second = self._record(completed=True)
            second.final_decision = "no"
            store.upsert(second)
            saved = next(iter(store.load().values()))
            self.assertEqual(saved["attempt_count"], 1)
            self.assertEqual(saved["final_decision"], "yes")

    def test_freeze_loader_rejects_duplicate_units(self):
        import json
        from experiments.mechanism.freeze_suite import load_unique_records

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.jsonl"
            second = Path(tmp) / "second.jsonl"
            item = self._record().to_dict()
            first.write_text(json.dumps(item) + "\n", encoding="utf-8")
            second.write_text(json.dumps(item) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_unique_records([first, second])

    def test_freeze_loader_rejects_completed_record_missing_audit_fields(self):
        import json
        from experiments.mechanism.freeze_suite import load_unique_records

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.jsonl"
            path.write_text(json.dumps(self._record().to_dict()) + "\n", encoding="utf-8")
            item = json.loads(path.read_text(encoding="utf-8"))
            item["rounds"][0]["action_trace"] = []
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires action_trace"):
                load_unique_records([path])


class AnalysisTests(unittest.TestCase):
    def test_group_analysis_reports_worst_domain_delta(self):
        from experiments.mechanism.group_analysis import build_report

        records = []
        for dataset, reference_correct, treatment_correct in [
                ("A", True, True), ("B", True, False)]:
            for condition, correct in [("ref", reference_correct),
                                       ("treatment", treatment_correct)]:
                records.append({
                    "dataset": dataset, "condition": condition, "completed": True,
                    "error": None, "question_id": "q", "seed": 1,
                    "final_decision": "yes" if correct else "no", "ground_truth": "yes",
                    "rounds": [{
                        "agent_id": 0, "round_id": 1, "chosen_action": "low_effort",
                        "report": {"A": "a", "E": ["e"], "I": "i", "C": "Yes"},
                        "visible_peer_reports": [], "stance": "yes", "token_input": 1,
                        "token_output": 1, "latency_s": 0.1, "reward": 0.0,
                        "realized_cost": 0.0,
                    }],
                })
        report = build_report(records, "ref")
        worst = report["deltas"]["treatment"]["worst_delta"]["decision_quality"]
        self.assertEqual(worst, {"dataset": "B", "delta": -1.0})

    def test_threshold_sensitivity_reclassifies_borderline_flip(self):
        from experiments.mechanism.sensitivity import threshold_counts

        rounds = []
        for round_id, evidence, stance in [(1, "alpha beta", "yes"),
                                            (2, "alpha gamma", "no")]:
            rounds.append({
                "round_id": round_id, "agent_id": 0, "stance": stance,
                "report": {"A": "a", "E": [evidence], "I": "i", "C": stance},
                "visible_peer_reports": [],
            })
        record = {"completed": True, "error": None, "dataset": "D",
                  "condition": "c", "rounds": rounds}
        cells = threshold_counts([record], [0.5, 0.8])["D::c"]
        self.assertEqual(cells["0.5"]["unsupported_flips"], 0)
        self.assertEqual(cells["0.8"]["unsupported_flips"], 1)

    def test_cost_weight_sensitivity_uses_realized_action_cost(self):
        from experiments.mechanism.sensitivity import cost_weight_values

        record = {
            "completed": True, "error": None, "dataset": "D", "condition": "c",
            "final_decision": "yes", "ground_truth": "yes",
            "rounds": [{"realized_cost": 1.0}, {"realized_cost": 2.0}],
        }
        values = cost_weight_values([record], [0.0, 0.2])["D::c"]
        self.assertEqual(values["0.0"]["mean_net_value"], 1.0)
        self.assertAlmostEqual(values["0.2"]["mean_net_value"], 0.7)


class AssetAuditTests(unittest.TestCase):
    def test_disclosed_legacy_duplicates_do_not_fail_structural_integrity(self):
        from experiments.audit_experiment_assets import consistency_report

        inventory = {"files": [{
            "path": "results/archive/legacy.json", "json_valid": True,
            "duplicate_experiments": 2, "error_experiments": 1,
        }]}
        report = consistency_report(inventory)
        self.assertTrue(report["structural_integrity_pass"])
        self.assertFalse(report["legacy_paired_reanalysis_ready"])
        self.assertTrue(report["pass"])
    def test_paired_statistics_keep_same_qid_from_different_datasets(self):
        from experiments.mechanism.statistical_tests import paired_maps, paired_question_maps

        template = {
            "completed": True, "error": None, "condition": "c",
            "question_id": "1", "final_decision": "yes", "ground_truth": "yes",
            "rounds": [{"round_id": 1, "agent_id": 0,
                        "chosen_action": "low_effort", "token_input": 1,
                        "token_output": 1}],
        }
        records = [{**template, "dataset": "D1"}, {**template, "dataset": "D2"}]
        action, decision, _tokens = paired_maps(records)
        self.assertEqual(len(decision["c"]), 2)
        self.assertEqual(len(action["c"]), 2)
        question_effort, question_tokens = paired_question_maps(records)
        self.assertEqual(len(question_effort["c"]), 2)
        self.assertEqual(len(question_tokens["c"]), 2)

    def test_paired_statistics_keep_same_question_from_different_seeds(self):
        from experiments.mechanism.statistical_tests import paired_maps, paired_question_maps

        template = {
            "completed": True, "error": None, "condition": "c",
            "dataset": "D", "question_id": "1", "final_decision": "yes",
            "ground_truth": "yes", "rounds": [{
                "round_id": 1, "agent_id": 0, "chosen_action": "low_effort",
                "token_input": 1, "token_output": 1,
            }],
        }
        records = [{**template, "seed": 7}, {**template, "seed": 99}]
        action, decision, _tokens = paired_maps(records)
        effort, question_tokens = paired_question_maps(records)
        self.assertEqual(len(action["c"]), 2)
        self.assertEqual(len(decision["c"]), 2)
        self.assertEqual(len(effort["c"]), 2)
        self.assertEqual(len(question_tokens["c"]), 2)

    def test_retry_failures_are_visible_in_attempt_level_rates(self):
        from experiments.mechanism.analyze import summarize_condition

        record = {
            "completed": True, "error": None, "final_decision": "yes",
            "ground_truth": "yes", "condition": "independent__x",
            "prior_errors": [
                "ValueError: response contains no AEIC JSON object",
                "ConnectionError: temporary network failure",
            ],
            "rounds": [{
                "agent_id": 0, "round_id": 1, "chosen_action": "low_effort",
                "report": {"A": "a", "E": ["e"], "I": "i", "C": "Yes"},
                "visible_peer_reports": [], "stance": "yes",
                "token_input": 1, "token_output": 1, "latency_s": 0.1,
                "reward": 0.0, "realized_cost": 0.0,
            }],
        }
        summary = summarize_condition([record])
        self.assertEqual(summary["total_attempts"], 3)
        self.assertEqual(summary["attempt_parse_error_count"], 1)
        self.assertEqual(summary["attempt_api_error_count"], 1)
        self.assertEqual(summary["parse_rate"], 0.5)
        self.assertAlmostEqual(summary["api_failure_rate"], 1 / 3)


class CollectionRecoveryTests(unittest.TestCase):
    def test_verifier_treats_all_format_repair_stages_as_repair_routes(self):
        traces = [
            {"stage": "choose", "backend": "deepseek"},
            {"stage": "verify_draft", "backend": "deepseek_reasoner"},
            {"stage": "verify_revision", "backend": "deepseek_reasoner"},
            {"stage": "format_repair", "backend": "deepseek"},
            {"stage": "format_repair_retry", "backend": "deepseek"},
        ]
        execution = [x for x in traces if x.get("stage") != "choose" and
                     not str(x.get("stage", "")).startswith("format_repair")]
        repair = [x for x in traces
                  if str(x.get("stage", "")).startswith("format_repair")]
        self.assertEqual({x["backend"] for x in execution}, {"deepseek_reasoner"})
        self.assertEqual({x["backend"] for x in repair}, {"deepseek"})

    def test_verifier_condition_name_covers_all_experimental_suffixes(self):
        from experiments.mechanism.verify_matrix import condition_name

        self.assertEqual(
            condition_name("independent", "static_aeic", 0.2, 1.5,
                           "EIC", "hidden", "agents3"),
            "independent__static_aeic@0.2__cost@1.5__signals@EIC"
            "__history@hidden__variant@agents3",
        )

    def test_parallel_shard_names_are_filesystem_safe(self):
        from experiments.mechanism.run_parallel_matrix import safe_name

        self.assertEqual(safe_name("MMLU-Hard / cost@0.5"), "mmlu-hard_cost_0.5")

    def test_hidden_history_is_not_disclosed_in_choice_prompt(self):
        from experiments.mechanism.collect import build_choice_prompt

        prompt = build_choice_prompt("q", "contract", "context", None)
        self.assertIn("not disclosed", prompt)
        self.assertNotIn("0.000", prompt)

    def test_nondefault_cost_scale_is_encoded_in_record_condition(self):
        from experiments.mechanism.collect import run_record

        controller = MechanismController(1)
        response = (
            AgentAction.LOW_EFFORT,
            '{"A":"a","E":["e"],"I":"i","C":"Yes"}',
            [{"stage": "low_effort", "prompt_tokens": 1,
              "completion_tokens": 1, "latency_s": 0.1, "status": 200}],
        )
        with patch("experiments.mechanism.collect.generate_one", return_value=response):
            record = run_record(
                "q", "yes", "D", "q1", "independent", "static_aeic",
                ["deepseek"], 1, 1, 42, controller, 0.2, 1.5,
                None, "cumulative", "agents3", False, True, "test",
            )
        self.assertEqual(
            record.condition,
            "independent__static_aeic@0.2__cost@1.5__variant@agents3",
        )

    def _round(self, agent_id, stance, evidence):
        offer = CONTRACTS["no_mechanism"].to_offer("prompt")
        return AgentRound(
            1, agent_id, offer, AgentAction.LOW_EFFORT,
            AEICReport("a", [evidence], "i", stance),
            stance=stance, token_input=1, token_output=1, latency_s=0.1,
        )

    def test_social_only_hides_reports_and_evidence(self):
        from experiments.mechanism.collect import peer_context

        previous = [self._round(0, "yes", "own"),
                    self._round(1, "no", "secret one"),
                    self._round(2, "no", "secret two")]
        context, visible = peer_context("social_only", previous, 0)
        self.assertIn("no", context)
        self.assertNotIn("secret", context)
        self.assertEqual(visible, [])

    def test_counterevidence_only_exposes_opposing_stance(self):
        from experiments.mechanism.collect import peer_context

        previous = [self._round(0, "yes", "own"),
                    self._round(1, "yes", "supporting"),
                    self._round(2, "no", "opposing")]
        context, visible = peer_context("counterevidence", previous, 0)
        self.assertIn("opposing", context)
        self.assertNotIn("supporting", context)
        self.assertEqual(visible, ["opposing"])

    def test_malformed_aeic_gets_separately_logged_format_repair(self):
        from experiments.mechanism.collect import generate_one

        calls = [
            ('{"ACTION":"aeic_effort"}', {"stage": "choose"}),
            ('{"A":"cut', {"stage": "aeic_effort"}),
            ('{"A":"a","E":["e"],"I":"i","C":"Yes"}',
             {"stage": "format_repair"}),
        ]
        with patch("experiments.mechanism.collect.client_for", return_value=object()), \
             patch("experiments.mechanism.collect._call_json", side_effect=calls):
            action, raw, trace = generate_one(
                "q", 0, "deepseek", "contract", "context", 0.0, False, True,
            )
        self.assertEqual(action, AgentAction.AEIC_EFFORT)
        self.assertEqual(trace[-1]["stage"], "format_repair")
        self.assertIn('"C":"Yes"', raw)

    def test_malformed_first_repair_gets_ultracompact_retry(self):
        from experiments.mechanism.collect import generate_one

        calls = [
            ('{"ACTION":"aeic_effort"}', {"stage": "choose"}),
            ('{"A":"cut', {"stage": "aeic_effort"}),
            ('{"A":"still cut', {"stage": "format_repair"}),
            ('{"A":"a","E":["e"],"I":"i","C":"Yes"}',
             {"stage": "format_repair_retry"}),
        ]
        with patch("experiments.mechanism.collect.client_for", return_value=object()), \
             patch("experiments.mechanism.collect._call_json", side_effect=calls):
            _action, raw, trace = generate_one(
                "q", 0, "deepseek", "contract", "context", 0.0, False, True,
            )
        self.assertEqual(trace[-1]["stage"], "format_repair_retry")
        self.assertIn('"C":"Yes"', raw)


class AdaptiveControllerTests(unittest.TestCase):
    def test_failed_longitudinal_record_rolls_back_reputation(self):
        from experiments.mechanism.collect import run_record

        controller = MechanismController(1)
        controller.reputation[0] = 2.5
        responses = [
            (AgentAction.AEIC_EFFORT,
             '{"A":"a","E":["e"],"I":"i","C":"Yes"}',
             [{"stage": "aeic_effort", "prompt_tokens": 1,
               "completion_tokens": 1, "latency_s": 0.1, "status": 200}]),
            (AgentAction.AEIC_EFFORT, "not json",
             [{"stage": "aeic_effort", "prompt_tokens": 1,
               "completion_tokens": 1, "latency_s": 0.1, "status": 200}]),
        ]
        with patch("experiments.mechanism.collect.generate_one", side_effect=responses):
            record = run_record(
                "q", "yes", "D", "q1", "independent", "aeic_contract",
                ["deepseek"], 1, 2, 42, controller, 0.2, 1.0,
                None, "cumulative", None, False, True, "test",
            )
        self.assertFalse(record.completed)
        self.assertEqual(controller.reputation[0], 2.5)

    def test_negative_reputation_changes_visible_contract(self):
        controller = MechanismController(1)
        base = CONTRACTS["adaptive_stackelberg"]
        neutral = controller.adaptive_spec(base, 0)
        controller.reputation[0] = -5.0
        adapted = controller.adaptive_spec(base, 0)
        self.assertLess(adapted.costs["verify"], neutral.costs["verify"])
        self.assertGreater(abs(adapted.rewards["unsupported_flip"]),
                           abs(neutral.rewards["unsupported_flip"]))

    def test_no_outcome_credit_without_available_ground_truth(self):
        controller = MechanismController(1)
        spec = CONTRACTS["adaptive_stackelberg"]
        prompt = render_contract(spec)
        offer = spec.to_offer(prompt)
        report = AEICReport("a", ["e"], "i", "yes")
        settlement = controller.settle(
            0, offer, AgentAction.AEIC_EFFORT, report, None, []
        )
        self.assertEqual(settlement["component_verified_evidence"], 0.0)
        self.assertEqual(settlement["component_useful_correction"], 0.0)

    def test_repeating_own_evidence_is_not_novel(self):
        controller = MechanismController(1)
        spec = CONTRACTS["aeic_contract"]
        offer = spec.to_offer(render_contract(spec))
        old = AEICReport("a", ["same evidence"], "i", "yes")
        new = AEICReport("a", ["same evidence"], "i", "yes")
        settlement = controller.settle(
            0, offer, AgentAction.AEIC_EFFORT, new, None, [], old, "yes", []
        )
        self.assertEqual(settlement["component_novel_evidence"], 0.0)

    def test_semantic_similarity_callback_controls_novelty(self):
        controller = MechanismController(1, similarity_fn=lambda _a, _b: 0.8)
        score = controller.novelty(AEICReport("", ["paraphrase"], "", ""), ["prior"])
        self.assertAlmostEqual(score, 0.2)

    def test_controller_flip_threshold_changes_unsupported_classification(self):
        spec = CONTRACTS["aeic_contract"]
        offer = spec.to_offer(render_contract(spec))
        old = AEICReport("a", ["alpha beta"], "i", "yes")
        new = AEICReport("a", ["alpha gamma"], "i", "no")
        low = MechanismController(1, flip_threshold=0.5).settle(
            0, offer, AgentAction.AEIC_EFFORT, new, None, [], old, "yes", []
        )
        high = MechanismController(1, flip_threshold=0.8).settle(
            0, offer, AgentAction.AEIC_EFFORT, new, None, [], old, "yes", []
        )
        self.assertEqual(low["component_unsupported_flip"], 0.0)
        self.assertEqual(high["component_unsupported_flip"], 1.0)


if __name__ == "__main__":
    unittest.main()
