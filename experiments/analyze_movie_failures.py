"""Mutually exclusive failure taxonomy for the frozen ReDial retrieval pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.game_consensus_recsys.domain import SearchRequest
from system.game_consensus_recsys.movie_data import HybridMovieCatalog


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path,
                        default=Path("experiments/benchmarks/movie_consensus_full/test.jsonl"))
    parser.add_argument("--catalog", type=Path,
                        default=Path("experiments/data/processed/movie_catalog.jsonl"))
    parser.add_argument("--associations", type=Path,
                        default=Path("experiments/data/processed/redial_train_cooccurrence.json"))
    parser.add_argument("--association-weight", type=float, default=0.85)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results/movie/failure_analysis.json"))
    parser.add_argument("--results", type=Path,
                        default=Path("experiments/results/movie/full_hybrid_w085.jsonl"),
                        help="frozen Top-K output used by the reported retrieval metrics")
    args = parser.parse_args()

    cases = load_jsonl(args.benchmark)
    frozen = {row["case_id"]: row for row in load_jsonl(args.results)}
    if set(frozen) != {case["case_id"] for case in cases}:
        raise ValueError("frozen result and benchmark case sets differ")
    catalog = HybridMovieCatalog(args.catalog, args.associations, args.association_weight)
    association_payload = json.loads(args.associations.read_text(encoding="utf-8"))
    association_keys = set(map(str, association_payload.get("associations", {})))
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    for case in cases:
        liked = list(map(str, case.get("liked_context_movie_ids", [])))
        relevant = set(map(str, case["relevant_movie_ids"]))
        request = SearchRequest(
            case["query"], top_k=5,
            excluded_movie_ids=list(map(str, case.get("excluded_seen_movie_ids", []))),
            liked_movie_ids=liked,
        )
        ranked = [item.movie_id for item in catalog.search(request, limit=50)]
        frozen_top5 = set(map(str, frozen[case["case_id"]]["ranked_movie_ids"][:5]))
        if relevant & frozen_top5:
            category = "hit_at_5"
        elif not liked:
            category = "no_pre_cutoff_liked_anchor"
        elif not (set(liked) & association_keys):
            category = "anchor_absent_from_train_associations"
        elif not (relevant & set(ranked)):
            category = "target_not_in_top_50"
        elif relevant & set(ranked[:10]):
            category = "target_in_top_10_but_not_top_5"
        else:
            category = "target_in_top_50_but_not_top_10"
        counts[category] += 1
        examples.setdefault(category, [])
        if len(examples[category]) < 5:
            examples[category].append(case["case_id"])

    order = [
        "hit_at_5", "no_pre_cutoff_liked_anchor",
        "anchor_absent_from_train_associations", "target_not_in_top_50",
        "target_in_top_50_but_not_top_10", "target_in_top_10_but_not_top_5",
    ]
    total = len(cases)
    report = {
        "cases": total,
        "association_weight": args.association_weight,
        "taxonomy_is_mutually_exclusive": sum(counts.values()) == total,
        "categories": [
            {"name": name, "count": counts[name], "rate": counts[name] / total,
             "example_case_ids": examples.get(name, [])}
            for name in order
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
