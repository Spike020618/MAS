"""Independent offline scorer. Never imported by the provider collector."""
import json
import math

from .core import CFG, OUT, sha, save, fingerprint, validate_rank
from .run import verify_plan


def score(ranking, positives):
    chosen = ranking[:CFG["top_k"]]
    hits = [mid in positives for mid in chosen]
    ideal = sum(1 / math.log2(k + 2) for k in range(min(len(positives), CFG["top_k"])))
    return {"hit_at_5": float(any(hits)), "recall_at_5": sum(hits) / len(positives) if positives else 0.0,
            "ndcg_at_5": sum(hit / math.log2(k + 2) for k, hit in enumerate(hits)) / ideal if ideal else 0.0}


def main():
    plan = verify_plan()
    if sha(OUT / "offline_labels.json") != plan["labels_sha256"]:
        raise ValueError("evaluation labels changed")
    cases = json.loads((OUT / "cases.json").read_text())
    labels = {x["case_id"]: set(x["positive_ids"]) for x in json.loads((OUT / "offline_labels.json").read_text())}
    per_case = []
    logical_slots = {"P0": [], "C0": [], "M00": [], "M10": ["P"], "M01": ["C"],
                     "M11": ["P", "C"], "U1": ["U1"], "U2": ["U1", "U2"]}
    outputs = []
    models = set()
    for case in cases:
        envelope = json.loads((OUT / "outputs" / (case["case_id"] + ".json")).read_text())
        output = envelope["record"]
        if envelope["dependency"] != fingerprint({"case": case, "plan": plan}) or envelope["integrity"] != fingerprint(output):
            raise ValueError("output integrity failure")
        for role, result in output["slots"].items():
            if sha(OUT / "slots" / case["case_id"] / (role + ".json")) != result["slot_sha256"]:
                raise ValueError("slot integrity failure")
            models.add(result["model_returned"])
        outputs.append(output)
        per_case.append({"case_id": case["case_id"], "positive_count": len(labels[case["case_id"]]),
                         "pool_positive_count": len(set(case["candidate_ids"]) & labels[case["case_id"]]),
                         "metrics": {name: score(validate_rank(rank, case["candidate_ids"]), labels[case["case_id"]])
                                     for name, rank in output["methods"].items()}})
    if len(models) != 1:
        raise ValueError("heterogeneous returned backend")
    aggregate = {}
    for name, roles in logical_slots.items():
        usage = [out["slots"][role] for out in outputs for role in roles]
        aggregate[name] = {metric: sum(c["metrics"][name][metric] for c in per_case) / len(cases)
                           for metric in ("hit_at_5", "recall_at_5", "ndcg_at_5")}
        aggregate[name].update(calls=len(usage), tokens=sum(s["usage"]["total_tokens"] for s in usage),
             input_tokens=sum(s["usage"]["prompt_tokens"] for s in usage),
             output_tokens=sum(s["usage"]["completion_tokens"] for s in usage),
             provider_seconds_sum=sum(s["elapsed_seconds"] for s in usage),
             invalid_reports=sum(s["error"] is not None for s in usage))
    transitions = {}
    for baseline in ("M00", "C0", "U1", "U2"):
        transitions[baseline] = {"corrected": sum(c["metrics"][baseline]["hit_at_5"] == 0 and c["metrics"]["M11"]["hit_at_5"] == 1 for c in per_case),
                                 "harmed": sum(c["metrics"][baseline]["hit_at_5"] == 1 and c["metrics"]["M11"]["hit_at_5"] == 0 for c in per_case)}
    save(OUT / "evaluation.json", {"scope": CFG["scope"], "cases": len(cases), "aggregate": aggregate,
         "M11_paired_hit_changes_against": transitions, "returned_models": sorted(models),
         "unique_provider_responses": sum(len(out["slots"]) for out in outputs),
         "unique_tokens": sum(s["usage"]["total_tokens"] for out in outputs for s in out["slots"].values()),
         "pool_hit_cases": sum(c["pool_positive_count"] > 0 for c in per_case), "per_case": per_case,
         "limitations": ["development engineering only; 8 previously opened users; no confirmatory inference",
                         "fixed RRF, no learned routing or game; no AgentCF reproduction claim",
                         "CF representation inherited; not independently retrained",
                         "different evidence realized per role, unified agent gets union",
                         "provider seconds are sums, not parallel user latency",
                         "all behavior labels are sparse observed future positives"]})
    print(json.dumps({"cases": len(cases), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
