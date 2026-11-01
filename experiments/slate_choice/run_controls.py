"""Run all collaborative dev controls, then label-blind exact-SMR diagnostics."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from .common import OUTPUT, PROTOCOL, sha, rows, write_once, sample_key
from .baselines import Recommender, score_prediction
from .controls import CFRecommender, capture_values, exact_smr
from .run_dev import interval
from .train_cf import CONTROL, CF_OUTPUT


def dependencies():
    files = {"protocol": PROTOCOL, "controls_protocol": CONTROL,
             "exact_limits": Path(__file__).with_name("exact_limits.json"),
             "data_manifest": OUTPUT / "data/manifest.json", "cf_manifest": CF_OUTPUT / "manifest.json",
             "baselines": Path(__file__).with_name("baselines.py"),
             "controls": Path(__file__).with_name("controls.py"),
             "runner": Path(__file__), "statistics_code": Path(__file__).with_name("run_dev.py"),
             "common": Path(__file__).with_name("common.py")}
    return {k: sha(v) for k, v in files.items()}


def run_cf(model, inputs, labels, priors, cfg, dep):
    output = OUTPUT / "controls/cf_development"
    write_once(output / "provenance.json", dep)
    completed = []
    for inp in inputs:
        for budget in cfg["history_budgets"]:
            path = output / "cases" / f"{inp['user_id']}_{budget}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["dependencies"] != dep:
                    raise ValueError("stale CF checkpoint")
                if record["metrics"] != score_prediction(record["prediction"], labels[inp["user_id"]], priors):
                    raise ValueError("CF checkpoint score changed")
            else:
                start = time.perf_counter()
                pred = model.predict(inp, budget)
                elapsed = time.perf_counter() - start
                metric = score_prediction(pred, labels[inp["user_id"]], priors)
                record = {"dependencies": dep, "prediction": pred, "metrics": metric,
                          "prediction_seconds": elapsed}
                write_once(path, record)
            completed.append(record)
            if len(completed) % 100 == 0:
                print(f"CF cases {len(completed)}/{len(inputs)*3}", flush=True)
    summary = {"dependencies": dep, "users": len(inputs), "cases": len(completed),
               "stage": "development; exploratory unadjusted intervals", "reserved_test_scored": False,
               "api_calls": 0, "budgets": {}}
    for budget in cfg["history_budgets"]:
        selected = [r for r in completed if r["prediction"]["history_budget"] == budget]
        item = {"users": len(selected), "methods": {}, "paired_recall_vs_point": {},
                "mean_candidate_size": float(np.mean([len(r["prediction"]["candidate_ids"]) for r in selected])),
                "candidate_hit_ceiling": float(np.mean([r["metrics"]["candidate_positive_count"] > 0 for r in selected]))}
        for method in cfg["methods"]:
            item["methods"][method] = {metric: float(np.mean([r["metrics"]["methods"][method][metric] for r in selected]))
                                       for metric in ("recall_at_5", "hit_at_5", "ndcg_at_5", "genre_distance", "observed_rating_coverage")}
            if method != "point_topk":
                diffs = [r["metrics"]["methods"][method]["recall_at_5"] -
                         r["metrics"]["methods"]["point_topk"]["recall_at_5"] for r in selected]
                item["paired_recall_vs_point"][method] = {
                    "mean": float(np.mean(diffs)), "bootstrap_95_percent_interval": interval(diffs, cfg["seed"] + budget),
                    "improved": sum(x > 0 for x in diffs), "harmed": sum(x < 0 for x in diffs)}
        summary["budgets"][str(budget)] = item
    write_once(output / "summary.json", summary)
    print("DONE collaborative development", flush=True)


def run_exact(models, inputs, labels, priors, cfg, ctrl, dep):
    output = OUTPUT / "controls/exact_diagnostic"
    write_once(output / "provenance.json", dep)
    limits = json.loads(Path(__file__).with_name("exact_limits.json").read_text())
    selected = sorted(inputs, key=lambda r: sample_key(r["user_id"]))[:ctrl["exact_max_users"]]
    target_cases = len(selected) * len(cfg["history_budgets"]) * len(models)
    completed, start = [], time.perf_counter()
    for inp in selected:
        for budget in cfg["history_budgets"]:
            for name, model in models.items():
                path = output / "cases" / f"{inp['user_id']}_{budget}_{name}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    if record["dependencies"] != dep:
                        raise ValueError("stale exact checkpoint")
                else:
                    if time.perf_counter() - start >= limits["total_wall_seconds"]:
                        print(f"exact wall guard reached; {len(completed)}/{target_cases}; resume command to continue", flush=True)
                        return
                    pred, matrices = capture_values(model, inp, budget)
                    original_ids = pred["predictions"]["greedy_minimax_regret"]["movie_ids"]
                    initial = [pred["candidate_ids"].index(mid) for mid in original_ids]
                    solved = exact_smr(matrices["values"], matrices["point"], cfg["top_k"], initial,
                                       ctrl["milp_time_limit_seconds_per_call"], limits["per_case_wall_seconds"],
                                       ctrl["solver_tolerance"])
                    exact_ids = [pred["candidate_ids"][j] for j in solved["chosen"]]
                    comparison = copy.deepcopy(pred)
                    comparison["predictions"] = {
                        "greedy_minimax_regret": pred["predictions"]["greedy_minimax_regret"],
                        "exact_minimax_regret": {"movie_ids": exact_ids}}
                    record = {"dependencies": dep, "representation": name,
                              "prediction": comparison, "solver": solved,
                              "metrics": score_prediction(comparison, labels[inp["user_id"]], priors)}
                    write_once(path, record)
                completed.append(record)
                if len(completed) % 20 == 0:
                    print(f"exact cases {len(completed)}/{target_cases}", flush=True)
    summary = {"dependencies": dep, "target_cases": target_cases, "completed": len(completed),
               "complete": len(completed) == target_cases, "users": len(selected),
               "reserved_test_scored": False, "api_calls": 0, "groups": {}}
    for name in models:
        for budget in cfg["history_budgets"]:
            subset = [r for r in completed if r["representation"] == name and r["prediction"]["history_budget"] == budget]
            summary["groups"][f"{name}_{budget}"] = {
                "cases": len(subset),
                "certified": sum(r["solver"]["global_optimum_certified_to_tolerance"] for r in subset),
                "objective_improved": sum(r["solver"]["initial_regret"] - r["solver"]["regret"] > ctrl["solver_tolerance"] for r in subset),
                "mean_initial_regret": float(np.mean([r["solver"]["initial_regret"] for r in subset])),
                "mean_final_regret": float(np.mean([r["solver"]["regret"] for r in subset])),
                "initial_zero_regret": sum(r["solver"]["initial_regret"] <= ctrl["solver_tolerance"] for r in subset),
                "final_zero_regret": sum(r["solver"]["regret"] <= ctrl["solver_tolerance"] for r in subset),
                "mean_solver_seconds": float(np.mean([r["solver"]["elapsed_seconds"] for r in subset])),
                "methods": {method: {metric: float(np.mean([r["metrics"]["methods"][method][metric] for r in subset]))
                                      for metric in ("recall_at_5", "hit_at_5", "ndcg_at_5")}
                            for method in ("greedy_minimax_regret", "exact_minimax_regret")}}
    write_once(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    cfg = json.loads(PROTOCOL.read_text())
    ctrl = json.loads(CONTROL.read_text())
    data = OUTPUT / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        if sha(data / name) != expected:
            raise ValueError("data changed")
    cf = json.loads((CF_OUTPUT / "manifest.json").read_text())
    if cf["dependencies"]["controls"] != sha(CONTROL) or sha(CF_OUTPUT / "features.npz") != cf["features_sha256"]:
        raise ValueError("CF source/config changed")
    if cf["dependencies"]["code"] != sha(Path(__file__).with_name("train_cf.py")):
        raise ValueError("CF trainer changed")
    priors = json.loads((data / "train_priors.json").read_text())["items"]
    inputs = rows(data / "dev.inputs.jsonl")
    labels = {r["user_id"]: r for r in rows(data / "dev.labels.jsonl")}
    collaborative = CFRecommender(priors, cfg, CF_OUTPUT / "features.npz")
    dep = dependencies()
    run_cf(collaborative, inputs, labels, priors, cfg, dep)
    run_exact({"genre": Recommender(priors, cfg), "collaborative": collaborative}, inputs, labels, priors, cfg, ctrl, dep)


if __name__ == "__main__":
    main()
