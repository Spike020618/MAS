"""Predeclared source-only development diagnostic. Never calls a model."""
import argparse
import json
import math
from pathlib import Path

from experiments.consensus_recommendation.common import ROOT, OUTPUT, create, fingerprint, read_rows, sha


def prepare(evidence_root, output):
    evidence_root, output = Path(evidence_root).resolve(), Path(output)
    manifest = json.loads((evidence_root / "manifest.json").read_text())
    if manifest["integrity"] != fingerprint({k: v for k, v in manifest.items() if k != "integrity"}):
        raise ValueError("evidence manifest changed")
    original = json.loads((OUTPUT / "data/manifest.json").read_text())
    plan = {"version": "training-signal-source-only-dev-v1", "evidence_root": str(evidence_root),
        "evidence_manifest_sha256": sha(evidence_root / "manifest.json"),
        "case_ids": sorted(manifest["cases"]), "expected_dev_labels_sha256": original["artifacts"]["dev.labels.jsonl"],
        "methods": ["frozen_retrieval", "context_score", "association_score"],
        "tie_policy": "descending raw signal, ties retain frozen candidate order; zero signal retains retrieval order",
        "metrics": ["Hit@5", "Recall@5", "NDCG@5"],
        "eligibility": "all fixed dev cases; empty target sets reported unannotated, not negative; pool misses remain in denominator",
        "scope": "development source-only diagnostics, not one of the five agent conditions and not evidence of MAS gain",
        "labels_read_during_prepare": False, "reserved_files_read": False, "new_api_calls": 0,
        "code_sha256": sha(Path(__file__))}
    plan["plan_hash"] = fingerprint(plan)
    create(output / "plan.json", plan)
    return {"status": "prepared_before_dev_scoring", "cases": len(plan["case_ids"]), "new_api_calls": 0}


def rankings(trace):
    rows = trace["candidate_signals"]
    return {"frozen_retrieval": [row["movie_id"] for row in rows[:5]],
        **{name: [row["movie_id"] for _, row in sorted(enumerate(rows), key=lambda pair: (-pair[1][field], pair[0]))[:5]]
           for name, field in (("context_score", "context_transfer_score"), ("association_score", "association_score"))}}


def metrics(ranking, targets):
    relevant = set(targets)
    if not relevant: return None
    hits = sum(mid in relevant for mid in ranking)
    dcg = sum(1 / math.log2(i+2) for i, mid in enumerate(ranking) if mid in relevant)
    ideal = sum(1 / math.log2(i+2) for i in range(min(5, len(relevant))))
    return {"Hit@5": int(hits > 0), "Recall@5": hits / len(relevant), "NDCG@5": dcg / ideal}


def score(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if plan["plan_hash"] != fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}) or sha(Path(__file__)) != plan["code_sha256"]:
        raise ValueError("diagnostic plan or code changed")
    evidence_root = Path(plan["evidence_root"])
    if sha(evidence_root / "manifest.json") != plan["evidence_manifest_sha256"]: raise ValueError("evidence changed")
    manifest = json.loads((evidence_root / "manifest.json").read_text())
    label_path = OUTPUT / "data/dev.labels.jsonl"
    if sha(label_path) != plan["expected_dev_labels_sha256"]: raise ValueError("development labels changed")
    # This is the first label read, after source generation and diagnostic freezing.
    labels = {r["case_id"]: r["relevant_movie_ids"] for r in read_rows(label_path)}
    rows = []
    for cid in plan["case_ids"]:
        trace = json.loads((evidence_root / "traces" / (cid.replace(":", "_") + ".json")).read_text())
        if fingerprint(trace) != manifest["cases"][cid]["trace_hash"]: raise ValueError("trace changed")
        targets = labels[cid]
        pool = trace["frozen_candidate_ids"]
        rows.append({"id": cid, "annotated": bool(targets), "pool_contains_target": bool(set(pool) & set(targets)),
                     "methods": {name: {"ranking": rank, "metrics": metrics(rank, targets)}
                                 for name, rank in rankings(trace).items()}})
    annotated = [row for row in rows if row["annotated"]]
    summaries = {}
    for name in plan["methods"]:
        summaries[name] = {metric: sum(row["methods"][name]["metrics"][metric] for row in annotated) / len(annotated)
                           for metric in plan["metrics"]}
        summaries[name]["hits"] = sum(row["methods"][name]["metrics"]["Hit@5"] for row in annotated)
    comparisons = {}
    for name in ("context_score", "association_score"):
        diff = [row["methods"][name]["metrics"]["Hit@5"] - row["methods"]["frozen_retrieval"]["metrics"]["Hit@5"]
                for row in annotated]
        comparisons[name] = {"wins": sum(d > 0 for d in diff), "losses": sum(d < 0 for d in diff),
                             "ties": sum(d == 0 for d in diff), "mean_hit_difference": sum(diff) / len(diff)}
    result = {"status": "development_only_scored", "plan_hash": plan["plan_hash"],
        "cases": len(rows), "annotated": len(annotated), "unannotated": len(rows) - len(annotated),
        "pool_has_target": sum(row["pool_contains_target"] for row in annotated),
        "summaries": summaries, "paired_against_frozen_retrieval": comparisons, "rows": rows,
        "dev_labels_read_for_scoring_only": True, "model_inputs_receive_dev_labels": False,
        "reserved_files_read": False, "new_api_calls": 0,
        "limits": ["development data already used in historical retrieval audits", "no independent test or significance claim",
                   "supervised retrieval signals do not establish current preferences or MAS benefit",
                   "do not select only winning development cases for future confirmatory evaluation"]}
    create(output / "score.json", result)
    return {k: v for k, v in result.items() if k != "rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    print(json.dumps(score(args.output) if args.score else prepare(args.evidence_root, args.output), ensure_ascii=False, indent=2))
