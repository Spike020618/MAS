"""Paired full-test comparison of lexical and train-only hybrid retrieval."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return {row["case_id"]: row for row in rows}


def metrics(row: dict, k: int = 5) -> tuple[float, float, float]:
    relevant = set(map(str, row["relevant_movie_ids"]))
    ranked = list(map(str, row["ranked_movie_ids"]))[:k]
    hits = sum(movie_id in relevant for movie_id in ranked)
    dcg = sum(1 / math.log2(index + 2) for index, movie_id in enumerate(ranked)
              if movie_id in relevant)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return float(hits > 0), hits / len(relevant), dcg / ideal


def exact_mcnemar(left: list[float], right: list[float]) -> dict:
    improved = sum(a == 0 and b == 1 for a, b in zip(left, right))
    degraded = sum(a == 1 and b == 0 for a, b in zip(left, right))
    discordant = improved + degraded
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, i) for i in range(min(improved, degraded) + 1))
        p_value = min(1.0, 2 * tail / (2 ** discordant))
    return {"hybrid_only_hits": improved, "lexical_only_hits": degraded,
            "discordant": discordant, "exact_two_sided_p": p_value}


def bootstrap_ci(differences: list[float], seed: int, repeats: int = 5000) -> list[float]:
    rng = random.Random(seed)
    size = len(differences)
    estimates = sorted(sum(differences[rng.randrange(size)] for _ in range(size)) / size
                       for _ in range(repeats))
    return [estimates[int(0.025 * repeats)], estimates[int(0.975 * repeats)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical", type=Path,
                        default=Path("experiments/results/movie/full_lexical.jsonl"))
    parser.add_argument("--hybrid", type=Path,
                        default=Path("experiments/results/movie/full_hybrid_w085.jsonl"))
    parser.add_argument("--cooccurrence", type=Path,
                        default=Path("experiments/results/movie/full_cooccurrence.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results/movie/retrieval_ablation.json"))
    parser.add_argument("--benchmark", type=Path,
                        default=Path("experiments/benchmarks/movie_consensus_full/test.jsonl"))
    args = parser.parse_args()
    lexical, hybrid, cooccurrence = load(args.lexical), load(args.hybrid), load(args.cooccurrence)
    benchmark = load(args.benchmark)
    if not (lexical.keys() == hybrid.keys() == cooccurrence.keys()):
        raise ValueError("paired result case sets differ")
    case_ids = sorted(lexical)
    left = [metrics(lexical[case_id]) for case_id in case_ids]
    right = [metrics(hybrid[case_id]) for case_id in case_ids]
    cooccurrence_values = [metrics(cooccurrence[case_id]) for case_id in case_ids]
    names = ("hit_at_5", "recall_at_5", "ndcg_at_5")
    fields = ("query", "relevant_movie_ids", "excluded_seen_movie_ids", "liked_context_movie_ids")
    inputs_match = all(
        case_id in benchmark
        and all(lexical[case_id].get(field) == benchmark[case_id].get(field)
                and hybrid[case_id].get(field) == benchmark[case_id].get(field)
                and cooccurrence[case_id].get(field) == benchmark[case_id].get(field)
                for field in fields)
        for case_id in case_ids
    )
    result = {"cases": len(case_ids), "paired_case_sets_equal": True,
              "benchmark_inputs_match": inputs_match, "metrics": {}}
    for index, name in enumerate(names):
        left_values = [row[index] for row in left]
        right_values = [row[index] for row in right]
        cooccurrence_metric = [row[index] for row in cooccurrence_values]
        differences = [b - a for a, b in zip(left_values, right_values)]
        result["metrics"][name] = {
            "lexical": sum(left_values) / len(left_values),
            "hybrid": sum(right_values) / len(right_values),
            "cooccurrence": sum(cooccurrence_metric) / len(cooccurrence_metric),
            "difference": sum(differences) / len(differences),
            "paired_bootstrap_95_ci": bootstrap_ci(differences, 20260907 + index),
        }
    result["hit_at_5_mcnemar"] = exact_mcnemar(
        [row[0] for row in left], [row[0] for row in right]
    )
    result["constraints"] = {
        "lexical": sum(bool(lexical[x]["constraint_satisfaction"]) for x in case_ids) / len(case_ids),
        "hybrid": sum(bool(hybrid[x]["constraint_satisfaction"]) for x in case_ids) / len(case_ids),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
