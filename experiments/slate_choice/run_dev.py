"""Run only development users; no option exists to score reserved test labels."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .baselines import Recommender, score_prediction
from .common import OUTPUT, PROTOCOL, sha, rows, write_once


def interval(values, seed):
    rng = np.random.default_rng(seed)
    a = np.array(values)
    boot = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)])
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUTPUT / "data")
    parser.add_argument("--output", type=Path, default=OUTPUT / "development")
    args = parser.parse_args()
    cfg = json.loads(PROTOCOL.read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        # Verify reserved artifact integrity but never deserialize its labels.
        if sha(args.data / name) != expected:
            raise ValueError(f"changed data: {name}")
    dependency = {"data_manifest": sha(args.data / "manifest.json"), "protocol": sha(PROTOCOL),
                  "runner": sha(Path(__file__)), "baselines": sha(Path(__file__).with_name("baselines.py")),
                  "common": sha(Path(__file__).with_name("common.py")),
                  "numpy_version": np.__version__}
    if manifest["dependencies"]["protocol"] != dependency["protocol"]:
        raise ValueError("protocol differs from built data")
    write_once(args.output / "provenance.json", dependency)
    priors = json.loads((args.data / "train_priors.json").read_text())
    inputs = rows(args.data / "dev.inputs.jsonl")
    if not inputs:
        raise ValueError("no eligible development users; cannot compute results")
    labels = {r["user_id"]: r for r in rows(args.data / "dev.labels.jsonl")}
    model = Recommender(priors["items"], cfg)
    input_hash = sha(args.data / "dev.inputs.jsonl")
    completed = []
    total = len(inputs) * len(cfg["history_budgets"])
    start = time.perf_counter()
    for inp in inputs:
        for budget in cfg["history_budgets"]:
            path = args.output / "cases" / f"{inp['user_id']}_{budget}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["dependencies"] != dependency or record["input_sha256"] != input_hash:
                    raise ValueError("stale case checkpoint")
                if (record["prediction"]["user_id"] != inp["user_id"] or
                        record["prediction"]["history_budget"] != budget or
                        record["metrics"] != score_prediction(record["prediction"], labels[inp["user_id"]], priors["items"])):
                    raise ValueError("invalid case checkpoint")
            else:
                tick = time.perf_counter()
                prediction = model.predict(inp, budget)
                elapsed = time.perf_counter() - tick
                # Evaluation sees labels only after the prediction is complete.
                metrics = score_prediction(prediction, labels[inp["user_id"]], priors["items"])
                record = {"dependencies": dependency, "input_sha256": input_hash,
                          "prediction": prediction, "metrics": metrics, "prediction_seconds": elapsed}
                write_once(path, record)
            completed.append(record)
            if len(completed) % 100 == 0:
                print(f"completed {len(completed)}/{total}; wall={time.perf_counter()-start:.1f}s", flush=True)
    report = {"dependencies": dependency, "users": len(inputs), "cases": len(completed),
              "stage": "development only; intervals exploratory and not multiplicity-adjusted",
              "reserved_test_evaluated": False, "model_calls": 0, "budgets": {}}
    for budget in cfg["history_budgets"]:
        selected = [r for r in completed if r["prediction"]["history_budget"] == budget]
        n = len(selected)
        item = {"users": n, "users_without_observed_future_positive": sum(r["metrics"]["future_positive_count"] == 0 for r in selected),
                "mean_candidate_size": float(np.mean([len(r["prediction"]["candidate_ids"]) for r in selected])),
                "candidate_hit_ceiling": float(np.mean([r["metrics"]["candidate_positive_count"] > 0 for r in selected])),
                "mean_candidate_positive_coverage": float(np.mean([r["metrics"]["candidate_positive_coverage"] for r in selected])),
                "mean_pipeline_prediction_seconds": float(np.mean([r["prediction_seconds"] for r in selected])),
                "methods": {}, "paired_recall_difference_vs_point": {}}
        for method in cfg["methods"]:
            item["methods"][method] = {metric: float(np.mean([r["metrics"]["methods"][method][metric] for r in selected]))
                                       for metric in ("hit_at_5", "recall_at_5", "ndcg_at_5", "genre_distance", "observed_rating_coverage")}
            if method != "point_topk":
                diffs = [r["metrics"]["methods"][method]["recall_at_5"] -
                         r["metrics"]["methods"]["point_topk"]["recall_at_5"] for r in selected]
                item["paired_recall_difference_vs_point"][method] = {
                    "mean": float(np.mean(diffs)), "bootstrap_95_percent_interval": interval(diffs, cfg["seed"] + budget),
                    "users_improved": sum(v > 0 for v in diffs), "users_harmed": sum(v < 0 for v in diffs)}
        report["budgets"][str(budget)] = item
    write_once(args.output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
