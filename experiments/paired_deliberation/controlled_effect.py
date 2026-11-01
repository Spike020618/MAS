"""Frozen atomic-v3 counterfactual mechanism experiment; not natural recommendation.

Scripted initial reports isolate updating. All five conditions see the full raw
evidence; no oracle labels or method-specific information restrictions at runtime.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, create, fingerprint, sha
from experiments.paired_deliberation.atomic_engine import check_release, run_suite
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.paired_deliberation.client_v5 import RawDeepSeek, parse_raw
from experiments.paired_deliberation.contracts_v6 import METHODS, OWNERS
from experiments.paired_deliberation.durable_provider import DurableProvider
from experiments.paired_deliberation.schema_echo_client import SchemaEchoClient
from experiments.paired_deliberation.state_v6 import StateV6
from experiments.paired_deliberation.validate_atomic_evidence import faithful

VALIDATION = ROOT / "experiments/results/paired_deliberation_d1_atomic_validation_r2"
FREEZE = ROOT / "experiments/results/paired_deliberation_d1_atomic_freeze_r1/freeze_manifest.json"


def claim(cid, text, refs, mids, kind="item_fact", stance="unknown", deps=None):
    return {"claim_id": cid, "text": text, "source_ids": refs, "candidate_ids": mids,
            "claim_type": kind, "stance": stance, "depends_on": deps or []}


def build():
    inputs, labels = [], []
    settings = [("Mystery", "Comedy", 0, 0, "positive_polarity"),
                ("Adventure", "Drama", 1, 1, "negative_polarity"),
                ("Musical", "Thriller", 2, 1, "utterance_to_state"),
                ("Western", "Romance", 0, 0, "irrelevant_year")]
    for world, (g, h, owner_index, position, family) in enumerate(settings):
        candidates = [{"source_id": "movie:%d" % i, "movie_id": str(i), "title": "Controlled Film %d" % i,
                       "year": 2000 + i, "genres": [g if i % 2 == 0 else h]} for i in range(1, 11)]
        # Candidate order is fixed by IDs, not by relevant targets or outcomes.
        candidates.sort(key=lambda c: fingerprint({"world": world, "movie": c["movie_id"]}))
        positive_ids = [c["movie_id"] for c in candidates if c["genres"] == [g]]
        negative_ids = [c["movie_id"] for c in candidates if c["genres"] == [h]]
        observations = ([{"source_id": "message:0", "role": "seeker", "text": f"I want {g} tonight."},
                         {"source_id": "message:1", "role": "seeker", "text": f"I do not want {g} tonight. "
                          "Both statements are simultaneous with equal standing; neither overrides the other."}]
                        if family == "utterance_to_state" else
                        [{"source_id": "message:0", "role": "seeker", "text": f"For tonight I want {g}. I do not want {h}. "
                          f"I like {g} and I do not like {h}."}])
        target_owner = OWNERS[owner_index]
        event_owner = OWNERS[(owner_index + 1) % 3]
        prediction_owner = OWNERS[(owner_index + 2) % 3]
        target_local = "c%d" % (position + 1)
        target = target_owner + ":" + target_local
        for corrupted in (False, True):
            case_id = "ce_" + fingerprint({"world": world, "variant": int(corrupted), "version": 1})[:12]
            evidence = {"case_id": case_id, "observations": copy.deepcopy(observations),
                "previous_mentions_excluded_not_asserted_seen": [],
                "candidates_in_shared_retrieval_order": copy.deepcopy(candidates)}
            reports = {}
            for idx, owner in enumerate(OWNERS):
                mid = str(idx + 1)
                facts = [claim("c1", f"Movie {mid} has catalog year {2000+idx+1}.", ["movie:" + mid], [mid]),
                         claim("c2", f"Movie {idx+4} has catalog year {2000+idx+4}.",
                               ["movie:%d" % (idx+4)], [str(idx+4)])]
                reports[owner] = {"ranking": [c["movie_id"] for c in candidates[:5]], "claims": facts, "updates": []}
            expected_atom = None
            if family == "positive_polarity":
                text = f"The seeker {'does not want' if corrupted else 'wants'} {g} tonight."
                target_claim = claim(target_local, text, ["message:0"], positive_ids,
                    "explicit_preference", "oppose" if corrupted else "support")
                expected_atom = {"scope": "preference_state", "subject": "seeker", "predicate": "wants",
                                 "object": g, "polarity": not corrupted, "time": "current"}
            elif family == "negative_polarity":
                text = f"The seeker currently {'likes' if corrupted else 'does not like'} {h}."
                target_claim = claim(target_local, text, ["message:0"], negative_ids,
                    "explicit_preference", "support" if corrupted else "oppose")
                expected_atom = {"scope": "preference_state", "subject": "seeker", "predicate": "likes",
                                 "object": h, "polarity": bool(corrupted), "time": "current"}
            elif family == "utterance_to_state":
                text = (f"The seeker wants {g} tonight." if corrupted else
                        f"In the supplied dialogue the seeker said they want {g} tonight.")
                target_claim = claim(target_local, text, ["message:0"], positive_ids, "explicit_preference", "support")
                expected_atom = {"scope": "preference_state" if corrupted else "utterance", "subject": "seeker",
                                 "predicate": "wants", "object": g, "polarity": True, "time": "current"}
            else:
                target_claim = claim(target_local, "Movie 8 has catalog year %d." % (2045 if corrupted else 2008),
                                     ["movie:8"], ["8"])
            reports[target_owner]["claims"][position] = target_claim
            event_text = (f"In the supplied dialogue the seeker said they do not want {g} tonight."
                          if family == "utterance_to_state" else
                          f"In the supplied dialogue the seeker said they want {g} tonight.")
            reports[event_owner]["claims"][1] = claim("c2", event_text,
                ["message:1"] if family == "utterance_to_state" else ["message:0"], positive_ids,
                "explicit_preference", "oppose" if family == "utterance_to_state" else "support")
            reports[prediction_owner]["claims"][1] = claim("c2", f"Films in {g} may fit the request.",
                ["message:0"], positive_ids, "recommendation_prediction", "support",
                deps=[event_owner + ":c2"] if family == "irrelevant_year" else [target])
            StateV6(reports, evidence)
            body = {"initial": reports, "calls": [], "evidence_hash": fingerprint(evidence)}
            shared = {**body, "integrity": fingerprint(body)}
            inputs.append({"id": case_id, "evidence": evidence, "shared": shared})
            protected = [owner + ":" + c["claim_id"] for owner, r in reports.items() for c in r["claims"]
                         if c["claim_type"] == "item_fact" and owner + ":" + c["claim_id"] != target]
            verdict = "supported" if not corrupted else "insufficient" if family == "utterance_to_state" else "refuted"
            labels.append({"id": case_id, "group": "world_%d" % world, "family": family, "corrupted": corrupted,
                "target": target, "target_claim": target_claim, "target_hash": fingerprint(target_claim),
                "expected_verdict": verdict, "expected_atom": expected_atom,
                "protected_registered_ids": protected,
                "compatible_ids": None if family == "utterance_to_state" else positive_ids,
                "decision_truth": "undefined under unresolved preference" if family == "utterance_to_state" else
                                  "explicit synthetic genre compatibility; not real choice or satisfaction"})
    return sorted(inputs, key=lambda r: r["id"]), sorted(labels, key=lambda r: r["id"])


def prepare(output):
    output = Path(output)
    _, validation = check_release(VALIDATION)
    frozen = json.loads(FREEZE.read_text())
    if frozen["integrity"] != fingerprint({k: v for k, v in frozen.items() if k != "integrity"}):
        raise ValueError("freeze manifest changed")
    for name, expected in frozen["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("frozen implementation changed")
    inputs, labels = build()
    create(output / "inputs.json", inputs)
    create(output / "labels.json", labels)
    files = set(frozen["code_sha256"]) | set(frozen["semantic_extra_code_sha256"]) | {str(Path(__file__).relative_to(ROOT))}
    plan = {"version": "atomic-v3-controlled-effect-r2-precollection-metrics", "status": "development_mechanism_experiment",
        "cases": 8, "groups": 4, "methods": list(METHODS), "rounds": 3,
        "model": "deepseek-chat", "temperature": 0, "expected_model": validation["physical"]["models"],
        "limits": frozen["limits"], "maximum_new_requests": 448,
        "request_bound": "8 * (independent1 + peer10 + central13 + verified_fixed16 + verified_adaptive16); scripted initial production has zero model calls",
        "inputs_hash": fingerprint(inputs), "labels_hash": fingerprint(labels),
        "freeze_sha256": sha(FREEZE), "validation_score_sha256": sha(VALIDATION / "release_score.json"),
        "code_sha256": {name: sha(ROOT / name) for name in sorted(files)},
        "runtime_labels_available": False, "scripted_initial_reports": True,
        "initial_cost_policy": "Scripted report production has no model calls for any arm; not an end-to-end cost comparison.",
        "intervention": "paired evidence/candidates/reports fixed except target text+stance and opaque case ID; target roles/positions fixed before inference",
        "primary_comparisons": [["verified_fixed", "peer"], ["verified_adaptive", "verified_fixed"], ["verified_adaptive", "central"]],
        "decision_metric": "synthetic explicit-preference compatible share@5 on 3 world pairs; conflict pair has no quality label",
        "process_metrics": ["known-original-target verifier coverage/accuracy/faithful interpretation", "false support on known target",
            "scoped-state outcomes", "registered literal retention (not semantic correction)", "dependencies", "rejections", "cost"],
        "failure_policy": "failed final path uses retrieval-top5 fallback in all-case decision metric; report success-only sensitivity separately; "
                          "invalid proposals remain recorded; unknown network billing or model change stops collection without retries",
        "analysis": "descriptive paired world results only; 4 groups do not support confirmatory significance or natural recommendation claims",
        "pilot_allowed": False, "reserved_test_allowed": False,
        "no_positive_result_gate": True, "stop": "one fixed batch; no parameter or sample changes after collection starts"}
    plan["plan_hash"] = fingerprint(plan)
    create(output / "plan.json", plan)
    for name in sorted(files):
        p, dest = ROOT / name, output / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != p.read_bytes(): raise ValueError("snapshot differs")
        dest.write_bytes(p.read_bytes())
    return {"status": "prepared", "cases": 8, "paths": 40, "groups": 4, "maximum_new_requests": 448,
            "new_api_calls": 0, "plan_hash": plan["plan_hash"]}


def verify_plan(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if plan["plan_hash"] != fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}): raise ValueError("plan changed")
    for name, expected in plan["code_sha256"].items():
        if sha(ROOT / name) != expected or sha(output / "source_snapshot" / name) != expected:
            raise ValueError("frozen source changed")
    if sha(FREEZE) != plan["freeze_sha256"] or sha(VALIDATION / "release_score.json") != plan["validation_score_sha256"]:
        raise ValueError("upstream acceptance changed")
    if fingerprint(json.loads((output / "inputs.json").read_text())) != plan["inputs_hash"]: raise ValueError("inputs changed")
    return plan


def active_rows(state):
    return {**state["supported_claims"], **state["uncertain_claims"]}


def score(output, plan):
    output = Path(output)
    labels = json.loads((output / "labels.json").read_text())
    if fingerprint(labels) != plan["labels_hash"]: raise ValueError("labels changed")
    inputs = {r["id"]: r for r in json.loads((output / "inputs.json").read_text())}
    details, summaries = [], {}
    for label in labels:
        saved = json.loads((output / "cases" / (label["id"] + ".json")).read_text())
        if saved["integrity"] != fingerprint(saved["result"]) or saved["result"]["plan_hash"] != plan["plan_hash"]:
            raise ValueError("case checkpoint changed")
        methods = saved["result"]["methods"]
        for method in METHODS:
            row = methods.get(method)
            completed = row is not None and row["status"] == "completed"
            result = row["result"] if completed else None
            ranking = result["final"]["ranking"] if completed else [c["movie_id"] for c in inputs[label["id"]]["evidence"]["candidates_in_shared_retrieval_order"][:5]]
            compatible = label["compatible_ids"]
            check = result["verification_cache"].get(label["target_hash"]) if completed else None
            state = result["state"] if completed else {"supported_claims": {}, "uncertain_claims": {}, "retracted_ids": []}
            active = active_rows(state)
            target_row = active.get(label["target"])
            status = (target_row["status"] if target_row else "retracted" if label["target"] in state["retracted_ids"] else "absent")
            initial = StateV6(inputs[label["id"]]["shared"]["initial"], inputs[label["id"]]["evidence"])
            protected_retained = sum(key in active and active[key]["claim"] == initial.claims[key]["claim"]
                                     for key in label["protected_registered_ids"])
            original_literal_active = None if not completed else any(
                r["claim"]["text"] == label["target_claim"]["text"] for r in active.values())
            scoped_outcome = None
            if completed and check is not None and method in {"verified_fixed", "verified_adaptive"}:
                scoped_outcome = (status == "supported" if label["expected_verdict"] == "supported" else
                                  not original_literal_active if label["expected_verdict"] == "refuted" else
                                  status == "unresolved" and original_literal_active)
            record = {"id": label["id"], "group": label["group"], "family": label["family"], "corrupted": label["corrupted"],
                "method": method, "completed": completed, "fallback_used": not completed,
                "ranking": ranking, "compatible_share_at5": None if compatible is None else len(set(ranking) & set(compatible)) / 5,
                "target_checked": check is not None, "target_check_correct": None if check is None else
                    check["verdict"] == label["expected_verdict"] and not check.get("verification_failed", False),
                "target_interpretation_faithful": None if check is None or label["expected_atom"] is None else
                    faithful(check.get("atom"), label["expected_atom"]),
                "target_false_support": check is not None and check["verdict"] == "supported" and label["expected_verdict"] != "supported",
                "target_false_refutation": check is not None and check["verdict"] == "refuted" and label["expected_verdict"] != "refuted",
                "target_status": status, "original_literal_active": original_literal_active,
                "scoped_state_outcome_matches_gold_conditional": scoped_outcome,
                "refutation_not_materialized": bool(completed and method in {"verified_fixed", "verified_adaptive"}
                    and check is not None and check["verdict"] == "refuted" and original_literal_active),
                "protected_exact_registered_retained": protected_retained if completed else None,
                "protected_registered_total": len(label["protected_registered_ids"]),
                "rejections": result["rejected"] if completed else [], "schedule": result["schedule"] if completed else [],
                "usage": result["usage"] if completed else row.get("usage") if row else None}
            details.append(record)
    for method in METHODS:
        rows = [r for r in details if r["method"] == method]
        valid_quality = [r for r in rows if r["compatible_share_at5"] is not None]
        checked = [r for r in rows if r["target_checked"]]
        pairs = []
        for group in sorted({r["group"] for r in rows}):
            clean, dirty = [next(r for r in rows if r["group"] == group and r["corrupted"] == flag) for flag in (False, True)]
            a, b = clean["compatible_share_at5"], dirty["compatible_share_at5"]
            pairs.append({"group": group, "family": clean["family"], "clean_quality": a, "corrupt_quality": b,
                "corrupt_minus_clean": None if a is None else b-a,
                "ranking_jaccard": len(set(clean["ranking"]) & set(dirty["ranking"])) / len(set(clean["ranking"]) | set(dirty["ranking"]))})
        usage = [r["usage"] for r in rows if r["usage"] is not None]
        summaries[method] = {"completed": sum(r["completed"] for r in rows), "cases": len(rows),
            "decision_cases": len(valid_quality), "compatible_share_at5_all_cases": sum(r["compatible_share_at5"] for r in valid_quality) / len(valid_quality),
            "compatible_share_at5_completed_only": (sum(r["compatible_share_at5"] for r in valid_quality if r["completed"]) /
                sum(r["completed"] for r in valid_quality)) if any(r["completed"] for r in valid_quality) else None,
            "target_checks": len(checked), "target_check_accuracy_conditional": sum(r["target_check_correct"] for r in checked) / len(checked) if checked else None,
            "target_joint_accuracy_conditional": (sum(r["target_check_correct"] and r["target_interpretation_faithful"] is not False
                for r in checked) / len(checked)) if checked else None,
            "target_false_supports": sum(r["target_false_support"] for r in rows),
            "target_false_refutations": sum(r["target_false_refutation"] for r in rows),
            "scoped_state_outcomes_checked": sum(r["scoped_state_outcome_matches_gold_conditional"] is not None for r in rows),
            "scoped_state_outcomes_correct": sum(r["scoped_state_outcome_matches_gold_conditional"] is True for r in rows),
            "refutations_not_materialized": sum(r["refutation_not_materialized"] for r in rows),
            "pairs": pairs, "logical_attempts": sum(u["attempts"] for u in usage),
            "known_prompt_tokens": sum(u["prompt_tokens"]["known"] for u in usage),
            "known_completion_tokens": sum(u["completion_tokens"]["known"] for u in usage),
            "usage_complete": all(u["usage_complete"] for u in usage),
            "rejections_by_kind": dict(Counter(e["kind"] for r in rows for e in r["rejections"]))}
    comparisons = []
    for a, b in plan["primary_comparisons"]:
        compared = [(x, next(r for r in details if r["id"] == x["id"] and r["method"] == b))
                    for x in details if x["method"] == a and x["compatible_share_at5"] is not None]
        deltas = [x["compatible_share_at5"] - y["compatible_share_at5"] for x, y in compared]
        comparisons.append({"methods": [a, b], "cases": len(deltas), "mean_delta": sum(deltas)/len(deltas),
                            "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas), "ties": sum(d == 0 for d in deltas)})
    result = {"status": "fixed_batch_scored", "plan_hash": plan["plan_hash"], "summaries": summaries,
        "primary_comparisons": comparisons, "details": details, "pilot_allowed": False,
        "limits": ["four synthetic worlds, scripted common initial reports", "no independent end-to-end single-agent claim",
                   "literal/registered retention is NOT semantic correction or factual preservation accuracy",
                   "conflict pair has no recommendation-quality labels", "descriptive paired results only"]}
    create(output / "score.json", result)
    return {k: v for k, v in result.items() if k != "details"}


def execute(output):
    output = Path(output)
    plan = verify_plan(output)
    check_release(VALIDATION)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    raw = RawDeepSeek(model=plan["model"])
    slots = output / "slots"
    class Transport:
        def count(self): return len(list(slots.glob("*.raw.json"))) + len(list(slots.glob("*.intent.json")))
        def preflight(self, request):
            raw.preflight(request)
            if self.count() >= plan["maximum_new_requests"]: raise RuntimeError("fixed batch request cap reached")
        def __call__(self, request):
            if self.count() > plan["maximum_new_requests"]: raise RuntimeError("fixed batch request cap reached")
            try: return raw(request)
            except OSError as exc: raise RuntimeError("network interrupted; no automatic repeat: " + str(exc)) from exc
    def parser(response, request):
        record = parse_raw(response, request)
        if record["model"] not in plan["expected_model"]: raise RuntimeError("returned model differs from frozen semantic release")
        return record
    provider = DurableProvider(slots, plan, Transport(), parser, True)
    complete = True
    for row in json.loads((output / "inputs.json").read_text()):
        path = output / "cases" / (row["id"] + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["integrity"] != fingerprint(saved["result"]) or saved["result"]["plan_hash"] != plan["plan_hash"]:
                raise ValueError("case checkpoint differs")
            if len(saved["result"]["methods"]) != len(METHODS): complete = False; break
            continue
        clients = {}
        def factory(method):
            clients[method] = SchemaEchoClient(AttemptClient(provider, plan["limits"]))
            return clients[method]
        methods = run_suite(row["evidence"], factory, VALIDATION, row["shared"], plan["rounds"])
        result = {"id": row["id"], "plan_hash": plan["plan_hash"], "methods": methods,
                  "metadata_echoes": {m: c.ignored_echoes for m, c in clients.items()}}
        create(path, {"result": result, "integrity": fingerprint(result)})
        print(json.dumps({"id": row["id"], "paths_completed": sum(r["status"] == "completed" for r in methods.values()),
                          "paths_attempted": len(methods)}), flush=True)
        if len(methods) != len(METHODS): complete = False; break
    if not complete:
        result = {"status": "incomplete_no_effect_conclusion", "plan_hash": plan["plan_hash"], "pilot_allowed": False}
        create(output / "incomplete.json", result)
        return result
    return score(output, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args.output) if args.execute_authorized else prepare(args.output), ensure_ascii=False, indent=2))
