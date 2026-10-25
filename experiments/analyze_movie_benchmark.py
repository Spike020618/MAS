"""Aggregate leakage-safe ReDial movie benchmark results."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from system.game_consensus_recsys.domain import SearchRequest
from system.game_consensus_recsys.movie_data import HybridMovieCatalog, JsonlMovieCatalog


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def case_metrics(row: dict, k: int = 5) -> dict[str, float]:
    relevant = set(map(str, row.get("relevant_movie_ids", [])))
    ranked = list(map(str, row.get("ranked_movie_ids", [])))[:k]
    hits = sum(movie_id in relevant for movie_id in ranked)
    dcg = sum(1.0 / math.log2(rank + 2) for rank, movie_id in enumerate(ranked)
              if movie_id in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {
        "hit_at_5": float(hits > 0),
        "recall_at_5": hits / len(relevant) if relevant else 0.0,
        "ndcg_at_5": dcg / ideal if ideal else 0.0,
    }


def exact_mcnemar(left: list[float], right: list[float]) -> dict:
    right_only = sum(a == 0 and b == 1 for a, b in zip(left, right))
    left_only = sum(a == 1 and b == 0 for a, b in zip(left, right))
    discordant = right_only + left_only
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, i)
                   for i in range(min(right_only, left_only) + 1))
        p_value = min(1.0, 2 * tail / (2 ** discordant))
    return {"right_only_hits": right_only, "left_only_hits": left_only,
            "discordant": discordant, "exact_two_sided_p": p_value}


def summarize(path: Path) -> dict:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"empty result file: {path}")
    per_case = [case_metrics(row) for row in rows]
    method = str(rows[0]["method"])
    expected_calls = {"lexical": 0, "single": 1, "ours": 3}[method]
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
    return {
        "method": method,
        "cases": len(rows),
        "hit_at_5": mean([x["hit_at_5"] for x in per_case]),
        "recall_at_5": mean([x["recall_at_5"] for x in per_case]),
        "ndcg_at_5": mean([x["ndcg_at_5"] for x in per_case]),
        "valid_entity_rate": 1.0 - mean([float(x.get("invalid_entity_rate", 0.0)) for x in rows]),
        "constraint_satisfaction_rate": mean([float(bool(x.get("constraint_satisfaction"))) for x in rows]),
        "evidence_support_rate": mean([float(x.get("evidence_support_rate", 0.0)) for x in rows]),
        "mean_tokens": mean([float(x.get("tokens", 0)) for x in rows]),
        "total_tokens": sum(int(x.get("tokens", 0)) for x in rows),
        "mean_model_calls": float(expected_calls),
        "total_model_calls": expected_calls * len(rows),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results/movie"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("experiments/benchmarks/movie_consensus/test.jsonl"))
    parser.add_argument("--catalog", type=Path,
                        default=Path("experiments/data/processed/movie_catalog.jsonl"))
    parser.add_argument("--retrieval", choices=["lexical", "hybrid"], default="lexical")
    parser.add_argument("--associations", type=Path,
                        default=Path("experiments/data/processed/redial_train_cooccurrence.json"))
    parser.add_argument("--association-weight", type=float, default=0.85)
    parser.add_argument("--result-prefix", default=None,
                        help="explicit filename prefix, e.g. hybrid_w085_")
    args = parser.parse_args()
    output = args.output or Path(
        "experiments/results/movie/summary_hybrid.json"
        if args.retrieval == "hybrid" else "experiments/results/movie/summary.json"
    )
    prefix = (args.result_prefix if args.result_prefix is not None else
              ("" if args.retrieval == "lexical" else f"{args.retrieval}_"))
    methods = [summarize(args.results_dir / f"{prefix}{name}.jsonl")
               for name in ("lexical", "single", "ours")]
    case_sets = [set(row["case_id"] for row in load_jsonl(args.results_dir / f"{prefix}{name}.jsonl"))
                 for name in ("lexical", "single", "ours")]
    catalog = (HybridMovieCatalog(args.catalog, args.associations, args.association_weight)
               if args.retrieval == "hybrid" else JsonlMovieCatalog(args.catalog))
    benchmark = load_jsonl(args.benchmark)
    benchmark_by_id = {row["case_id"]: row for row in benchmark}
    input_fields = ("query", "relevant_movie_ids", "excluded_seen_movie_ids",
                    "liked_context_movie_ids")
    benchmark_inputs_match = True
    for name in ("lexical", "single", "ours"):
        for row in load_jsonl(args.results_dir / f"{prefix}{name}.jsonl"):
            source = benchmark_by_id.get(row["case_id"])
            if source is None or any(row.get(field) != source.get(field) for field in input_fields):
                benchmark_inputs_match = False
    retrieval_hits = {10: [], 50: []}
    retrieval_recalls = {10: [], 50: []}
    for case in benchmark:
        request = SearchRequest(
            case["query"], top_k=5,
            excluded_movie_ids=list(map(str, case.get("excluded_seen_movie_ids", []))),
            liked_movie_ids=list(map(str, case.get("liked_context_movie_ids", []))),
        )
        pool = catalog.search(request, limit=50)
        relevant = set(map(str, case["relevant_movie_ids"]))
        for cutoff in (10, 50):
            found = relevant & {candidate.movie_id for candidate in pool[:cutoff]}
            retrieval_hits[cutoff].append(float(bool(found)))
            retrieval_recalls[cutoff].append(len(found) / len(relevant))
    result_rows = {
        name: {row["case_id"]: row for row in load_jsonl(args.results_dir / f"{prefix}{name}.jsonl")}
        for name in ("lexical", "single", "ours")
    }
    ordered_ids = sorted(case_sets[0])
    hit_vectors = {
        name: [case_metrics(result_rows[name][case_id])["hit_at_5"] for case_id in ordered_ids]
        for name in result_rows
    }
    report = {
        "benchmark": "ReDial-to-MovieLens leakage-safe test split",
        "retrieval": args.retrieval,
        "top_k": 5,
        "paired_case_sets_equal": case_sets[0] == case_sets[1] == case_sets[2],
        "benchmark_inputs_match": benchmark_inputs_match,
        "retrieval_ceiling": {
            f"hit_at_{cutoff}": mean(retrieval_hits[cutoff])
            for cutoff in (10, 50)
        } | {
            f"recall_at_{cutoff}": mean(retrieval_recalls[cutoff])
            for cutoff in (10, 50)
        },
        "methods": methods,
        "paired_hit_at_5_tests": {
            "ours_vs_direct": exact_mcnemar(hit_vectors["lexical"], hit_vectors["ours"]),
            "ours_vs_single": exact_mcnemar(hit_vectors["single"], hit_vectors["ours"]),
            "single_vs_direct": exact_mcnemar(hit_vectors["lexical"], hit_vectors["single"]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
