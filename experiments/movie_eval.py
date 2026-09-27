"""Offline movie recommendation metrics and question-level paired bootstrap."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(relevant: set[str], ranked: list[str], k: int = 5) -> dict[str, float]:
    top = ranked[:k]
    hits = [1.0 if movie_id in relevant else 0.0 for movie_id in top]
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return {
        f"hit@{k}": float(any(hits)),
        f"recall@{k}": sum(hits) / len(relevant) if relevant else 0.0,
        f"ndcg@{k}": dcg / ideal if ideal else 0.0,
    }


def summarize(records: list[dict], k: int = 5) -> dict:
    rows = []
    for record in records:
        values = metrics(set(map(str, record["relevant_movie_ids"])),
                         list(map(str, record.get("ranked_movie_ids", []))), k)
        values.update({
            "constraint_satisfaction": float(record.get("constraint_satisfaction", True)),
            "invalid_entity_rate": float(record.get("invalid_entity_rate", 0.0)),
            "evidence_support_rate": float(record.get("evidence_support_rate", 0.0)),
            "tokens": float(record.get("tokens", 0)), "latency_ms": float(record.get("latency_ms", 0)),
        })
        rows.append(values)
    keys = sorted(rows[0]) if rows else []
    return {"n": len(rows), **{key: sum(row[key] for row in rows) / len(rows) for key in keys}}


def paired_bootstrap(left: dict[str, dict], right: dict[str, dict], endpoint: str,
                     iterations: int = 10000, seed: int = 20260907) -> dict:
    common = sorted(set(left) & set(right))
    diffs = [right[key][endpoint] - left[key][endpoint] for key in common]
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        samples.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
    samples.sort()
    return {"n_pairs": len(diffs), "mean_delta": sum(diffs) / len(diffs),
            "ci95": [samples[int(0.025 * iterations)], samples[int(0.975 * iterations) - 1]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(load(args.prediction), args.k)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
