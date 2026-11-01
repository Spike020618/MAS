"""Offline development scoring. No imports from this file in model execution."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.consensus_recommendation.common import OUTPUT, ROOT, create, fingerprint, read_rows, sha
from experiments.consensus_recommendation.evaluate import metrics, paired


def score(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    labels = {row["case_id"]: row for row in read_rows(OUTPUT / "data/dev.labels.jsonl")}
    raw = {"redial:" + str(row["conversationId"]): str(row["initiatorWorkerId"])
           for row in read_rows(ROOT / "experiments/data/raw/redial/train.jsonl")}
    cases, missing = [], []
    artifact_hashes = {}
    for cid in plan["case_ids"]:
        path = root / "cases" / (cid.replace(":", "_") + ".json")
        if not path.exists():
            missing.append(cid)
            continue
        envelope = json.loads(path.read_text())
        if envelope["integrity"] != fingerprint(envelope["record"]):
            raise ValueError("case integrity failed: " + cid)
        cases.append(envelope["record"])
        artifact_hashes[str(path.relative_to(root))] = sha(path)
    if not cases:
        raise ValueError("no completed cases")
    values, validity, costs, events, per_case = defaultdict(list), Counter(), defaultdict(list), Counter(), []
    initial_diagnostics = Counter()
    invalid_reasons = Counter()
    for case in cases:
        cid = case["case_id"]
        row = {"case_id": cid, "methods": {}}
        for family in ("E1", "E2"):
            for method, entry in case[family].items():
                name = family + ":" + method
                valid = entry["status"] == "valid"
                result = entry.get("result", {})
                rank = result.get("decision", {}).get("ranking", []) if valid else []
                measure = metrics(rank, labels[cid]["relevant_movie_ids"])
                validity[name] += valid
                if not valid:
                    invalid_reasons[name + ":" + entry.get("error", "unknown")] += 1
                values[name].append(measure)
                row["methods"][name] = {"valid": valid, **measure, "ranking": rank}
                if valid:
                    costs[name].append(result["ledger"])
                    for event in result.get("events", []):
                        events[name + ":" + event["decision"]] += 1
                        events[name + ":state_changed"] += event["before"] != event["after"]
        per_case.append(row)
        update_result = case["E2"]["evidence_update"].get("result")
        if update_result:
            initial = update_result["initial_reports"]
            gold = set(labels[cid]["relevant_movie_ids"])
            union_hit = any(gold.intersection(report["ranking"][:5]) for report in initial.values())
            final_hit = bool(gold.intersection(update_result["decision"].get("ranking", [])[:5]))
            initial_diagnostics["cases_with_initial_reports"] += 1
            initial_diagnostics["initial_union_hit"] += union_hit
            initial_diagnostics["initial_union_hit_lost"] += union_hit and not final_hit
            initial_diagnostics["new_hit_outside_initial_top5_union"] += final_hit and not union_hit
    n = len(cases)
    methods = {}
    for name, rows in sorted(values.items()):
        ledgers = costs[name]
        methods[name] = {
            "completed_cases": n, "valid": validity[name],
            "hit_count": sum(row["hit"] for row in rows),
            "hit_execution_lower": sum(row["hit"] for row in rows) / n,
            "hit_upper_invalid_as_success": (sum(row["hit"] for row in rows) + n - validity[name]) / n,
            "planned_denominator_hit_lower": sum(row["hit"] for row in rows) / len(plan["case_ids"]),
            "planned_denominator_hit_upper": (sum(row["hit"] for row in rows)
                + len(plan["case_ids"]) - validity[name]) / len(plan["case_ids"]),
            "recall_at_5": sum(row["recall"] for row in rows) / n,
            "ndcg_at_5": sum(row["ndcg"] for row in rows) / n,
            "valid_path_prompt_tokens": sum(row["prompt_tokens"] for row in ledgers),
            "valid_path_completion_tokens": sum(row["completion_tokens"] for row in ledgers),
        }
    groups = [raw[case["case_id"]] for case in cases]
    comparisons = {}
    target = "E2:evidence_update"
    for reference in ("E2:peer_review", "E2:self_review", "E2:independent", "E1:single_serial"):
        comparisons[target + "_vs_" + reference] = paired(
            [row["hit"] for row in values[target]],
            [row["hit"] for row in values[reference]], groups,
        )
    slots, tokens, integrity = 0, Counter(), Counter()
    models = Counter()
    for path in sorted((root / "slots").glob("*.json")):
        if path.name.endswith(".intent.json"):
            continue
        envelope = json.loads(path.read_text())
        record = envelope["record"]
        if envelope["integrity"] != fingerprint(record):
            raise ValueError("slot integrity failed: " + str(path))
        slots += 1
        models[record["model"]] += 1
        for key in ("prompt_tokens", "completion_tokens"):
            tokens[key] += record["usage"][key]
        integrity.update(record.get("integrity_issues", []))
        artifact_hashes[str(path.relative_to(root))] = sha(path)
    intents = [str(path.relative_to(root)) for path in (root / "slots").glob("*.intent.json")
               if not path.with_name(path.name.replace(".intent.json", ".json")).exists()]
    complete = not missing and not intents
    direction = all(comparisons[target + "_vs_" + ref]["difference"] > 0
                    for ref in ("E2:peer_review", "E1:single_serial"))
    return {
        "version": "paired-development-scoring-v1", "plan_sha256": sha(root / "plan.json"),
        "planned_cases": len(plan["case_ids"]), "completed_cases": n,
        "missing_cases": missing, "unresolved_intents": intents, "collection_complete": complete,
        "methods": methods, "paired_development_comparisons": comparisons,
        "positive_direction_on_completed_cases": direction,
        "formal_expansion_allowed": False,
        "reason": "Development results require completed collection and mechanism-validity review before expansion.",
        "provider_usage": {"successful_saved_responses": slots, **dict(tokens),
                           "returned_models": dict(models), "actual_cost": None},
        "integrity_issue_counts": dict(integrity), "state_event_counts": dict(events),
        "invalid_path_reasons": dict(invalid_reasons),
        "initial_complementarity_diagnostics": dict(initial_diagnostics),
        "process_semantic_evaluation": "not_run; state change does not certify semantic correction",
        "controlled_error_method_experiment": "not_run; 288 generated cases are fixtures only",
        "e3_execution": "not_run; specifications are not executable experimental arms",
        "reserved_test_read": False, "new_api_calls_for_scoring": 0,
        "artifact_sha256": artifact_hashes, "per_case": per_case,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.root)
    create(args.output, report)
    print(json.dumps({key: report[key] for key in (
        "planned_cases", "completed_cases", "collection_complete", "methods", "provider_usage",
        "positive_direction_on_completed_cases")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
