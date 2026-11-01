"""Prepare a label-free, method-blind claim annotation packet."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.consensus_recommendation.common import OUTPUT, ROOT, create, read_rows


def blind_id(case_id, qualified_claim_id):
    raw = (case_id + "\0" + qualified_claim_id).encode()
    return "claim-" + hashlib.sha256(raw).hexdigest()[:16]


def prepare(stage="pilot", limit=12):
    base = OUTPUT / "data"
    inputs = {row["case_id"]: row for row in read_rows(base / "dev.inputs.jsonl")}
    candidates = {row["case_id"]: row for row in read_rows(base / "dev.candidates.jsonl")}
    _, _, movies = __import__(
        "experiments.scripts.data.build_redial_benchmark", fromlist=["load_catalog"]
    ).load_catalog(ROOT / "experiments/data/processed/movie_catalog.jsonl")
    case_root = OUTPUT / "collection_execution_v3" / stage / "cases"
    cases = []
    mapping = []
    for path in sorted(case_root.glob("*.json"), key=lambda item: item.name)[:limit]:
        record = json.loads(path.read_text())["record"]
        case_id = record["case_id"]
        claims = []
        for role, report in sorted(record["initial_reports"].items()):
            for claim in report["claims"]:
                qualified = role + ":" + claim["claim_id"]
                annotation_id = blind_id(case_id, qualified)
                claims.append({
                    "annotation_id": annotation_id,
                    "claim_text": claim["text"],
                    "stance": claim["stance"],
                    "candidate_ids": claim["candidate_ids"],
                    "cited_source_ids": claim["source_ids"],
                    "annotation": None,
                })
                mapping.append({"annotation_id": annotation_id, "case_id": case_id,
                                "qualified_claim_id": qualified})
        movie_rows = []
        for movie_id in candidates[case_id]["ranked_movie_ids"]:
            movie = movies[movie_id]
            movie_rows.append({"source_id": "movie:" + movie_id, "movie_id": movie_id,
                               "title": movie["title"], "year": movie.get("year"),
                               "genres": movie.get("genres", [])})
        cases.append({
            "annotation_case_id": "case-" + hashlib.sha256(case_id.encode()).hexdigest()[:12],
            "observations": inputs[case_id]["observations"],
            "candidate_facts": movie_rows,
            "claims": claims,
        })
    return {
        "version": "paired-deliberation.blind-annotation-packet-v1",
        "stage": stage,
        "cases": cases,
        "instructions": {
            "labels": ["supported", "unsupported", "unclear"],
            "independent_annotators_required": 2,
            "method_arm_hidden": True,
            "final_recommendations_hidden": True,
            "evaluation_labels_hidden": True,
            "do_not_infer": "A valid source ID does not imply that it supports the claim."
        },
        "private_mapping": mapping,
        "reserved_labels_read": False,
        "new_model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("engineering", "pilot"), default="pilot")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    result = prepare(args.stage, args.limit)
    create(args.output, result)
    print(json.dumps({"cases": len(result["cases"]),
                      "claims": sum(len(case["claims"]) for case in result["cases"]),
                      "reserved_labels_read": False, "new_model_calls": 0}))


if __name__ == "__main__":
    main()
