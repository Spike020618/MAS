"""Paired question-level cost-effectiveness with cluster bootstrap intervals."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def question_metrics(record: dict) -> dict:
    rounds = record.get("rounds") or []
    return {
        "correct": float(record.get("final_decision") == record.get("ground_truth")),
        "high_effort": sum(x.get("chosen_action") in {"aeic_effort", "verify"} for x in rounds) / len(rounds),
        "tokens": sum((x.get("token_input") or 0) + (x.get("token_output") or 0) for x in rounds),
        "calls": sum(len(x.get("action_trace") or []) for x in rounds),
        "latency_s": sum(float(x.get("latency_s") or 0) for x in rounds),
    }


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def paired_summary(pairs: list[tuple[dict, dict]], samples: int, seed: int) -> dict:
    endpoints = ("correct", "high_effort", "tokens", "calls", "latency_s")
    differences = {name: [right[name] - left[name] for left, right in pairs]
                   for name in endpoints}
    rng = random.Random(seed)
    boot = {name: [] for name in endpoints}
    for _ in range(samples):
        indices = [rng.randrange(len(pairs)) for _ in pairs]
        for name in endpoints:
            boot[name].append(mean(differences[name][i] for i in indices))
    result = {"n_pairs": len(pairs), "endpoints": {}}
    for name in endpoints:
        result["endpoints"][name] = {
            "left_mean": mean(x[0][name] for x in pairs),
            "right_mean": mean(x[1][name] for x in pairs),
            "delta": mean(differences[name]),
            "ci95": [percentile(boot[name], .025), percentile(boot[name], .975)],
        }
    net_corrections = sum(right["correct"] - left["correct"] for left, right in pairs)
    extra_tokens = sum(right["tokens"] - left["tokens"] for left, right in pairs)
    extra_calls = sum(right["calls"] - left["calls"] for left, right in pairs)
    result["net_corrections"] = net_corrections
    result["extra_tokens_total"] = extra_tokens
    result["extra_calls_total"] = extra_calls
    result["tokens_per_net_correction"] = (extra_tokens / net_corrections
                                            if net_corrections > 0 else None)
    result["calls_per_net_correction"] = (extra_calls / net_corrections
                                           if net_corrections > 0 else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--left", default="independent__no_mechanism@0.2")
    parser.add_argument("--right", default="independent__static_aeic@0.2")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    index = defaultdict(dict)
    for path in args.records:
        for record in load(path):
            if record.get("completed") and not record.get("error"):
                index[(record["dataset"], str(record["question_id"]))][record["condition"]] = question_metrics(record)
    by_dataset = defaultdict(list)
    pooled = []
    for (dataset, _qid), values in index.items():
        if args.left in values and args.right in values:
            pair = (values[args.left], values[args.right])
            by_dataset[dataset].append(pair)
            pooled.append(pair)
    result = {
        "left": args.left, "right": args.right,
        "bootstrap_samples": args.bootstrap_samples, "seed": args.seed,
        "overall": paired_summary(pooled, args.bootstrap_samples, args.seed),
        "by_dataset": {name: paired_summary(pairs, args.bootstrap_samples, args.seed + i + 1)
                       for i, (name, pairs) in enumerate(sorted(by_dataset.items()))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
