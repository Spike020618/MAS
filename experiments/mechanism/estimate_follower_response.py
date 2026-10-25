"""Estimate the finite-contract follower response and select a dev-optimal contract.

The script deliberately separates two claims:
1. a descriptive logistic response surface for follower high-resource actions;
2. exact enumeration of the observed finite contract grid for leader selection.
It never labels the result a closed-form or unrestricted Stackelberg equilibrium.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


HIGH_RESOURCE = {"aeic_effort", "verify"}
CONDITION_RE = re.compile(
    r"^independent__static_aeic@(?P<reward>[0-9.]+)(?:__cost@(?P<cost>[0-9.]+))?$"
)


def load_jsonl(paths: list[Path]) -> list[dict]:
    latest: dict[tuple, dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("experiment_id"), row.get("dataset"), row.get("condition"),
                   str(row.get("question_id")), row.get("seed"))
            latest[key] = row
    return list(latest.values())


def parse_contract(condition: str) -> tuple[float, float]:
    match = CONDITION_RE.match(condition)
    if not match:
        raise ValueError(f"unexpected condition for response grid: {condition}")
    return float(match.group("reward")), float(match.group("cost") or 1.0)


def minmax(value: float, values: list[float]) -> float:
    low, high = min(values), max(values)
    return 0.0 if math.isclose(low, high) else (value - low) / (high - low)


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in records:
        if row.get("completed") is not True or row.get("error"):
            continue
        groups[parse_contract(str(row["condition"]))].append(row)
    grid = []
    for (reward, cost), rows in sorted(groups.items()):
        rounds = [item for row in rows for item in row.get("rounds", [])]
        tokens = [float((x.get("token_input") or 0) + (x.get("token_output") or 0))
                  for x in rounds]
        calls = [sum(1 for trace in (x.get("action_trace") or [])
                     if trace.get("stage") != "choose") for x in rounds]
        latencies = [float(x.get("latency_s") or 0.0) for x in rounds]
        question_latencies = []
        for row in rows:
            by_round: dict[int, list[float]] = defaultdict(list)
            for item in row.get("rounds", []):
                by_round[int(item.get("round_id", 0))].append(float(item.get("latency_s") or 0.0))
            # Agents execute concurrently within a round and rounds execute sequentially.
            question_latencies.append(sum(max(values) for values in by_round.values() if values))
        failures = sum(1 for row in rows if row.get("completed") is not True or row.get("error"))
        grid.append({
            "contract_id": f"r{reward:g}_c{cost:g}",
            "reward_scale": reward,
            "cost_scale": cost,
            "n_questions": len(rows),
            "n_agent_rounds": len(rounds),
            "accuracy": sum(row.get("final_decision") == row.get("ground_truth") for row in rows) / len(rows),
            "high_resource_rate": sum(x.get("chosen_action") in HIGH_RESOURCE for x in rounds) / len(rounds),
            "mean_tokens_per_agent_round": float(np.mean(tokens)),
            "mean_execution_calls_per_agent_round": float(np.mean(calls)),
            "mean_latency_s_per_agent_round": float(np.mean(latencies)),
            "p95_latency_s_per_agent_round": float(np.percentile(latencies, 95)),
            "mean_estimated_question_latency_s": float(np.mean(question_latencies)),
            "p95_estimated_question_latency_s": float(np.percentile(question_latencies, 95)),
            "failure_rate": failures / len(rows),
        })
    if not grid:
        raise ValueError("no completed grid records")
    return grid


def response_model(records: list[dict]) -> dict:
    domains = sorted({str(row["dataset"]) for row in records})
    baseline = domains[0]
    X, y = [], []
    for row in records:
        if row.get("completed") is not True or row.get("error"):
            continue
        reward, cost = parse_contract(str(row["condition"]))
        domain = str(row["dataset"])
        features = [reward, cost, reward * cost]
        features.extend(1.0 if domain == name else 0.0 for name in domains[1:])
        for item in row.get("rounds", []):
            X.append(features)
            y.append(int(item.get("chosen_action") in HIGH_RESOURCE))
    model = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs").fit(np.asarray(X), np.asarray(y))
    names = ["reward", "cost", "reward_x_cost"] + [f"domain_{x}" for x in domains[1:]]
    return {
        "model": "descriptive agent-round logistic regression; coefficients are not cluster-robust",
        "baseline_domain": baseline,
        "n_agent_rounds": len(y),
        "intercept": float(model.intercept_[0]),
        "coefficients": {name: float(value) for name, value in zip(names, model.coef_[0])},
        "training_accuracy": float(model.score(np.asarray(X), np.asarray(y))),
    }


def pareto_frontier(grid: list[dict]) -> list[str]:
    result = []
    for candidate in grid:
        dominated = any(
            other is not candidate
            and other["accuracy"] >= candidate["accuracy"]
            and other["mean_tokens_per_agent_round"] <= candidate["mean_tokens_per_agent_round"]
            and other["mean_execution_calls_per_agent_round"] <= candidate["mean_execution_calls_per_agent_round"]
            and (other["accuracy"] > candidate["accuracy"]
                 or other["mean_tokens_per_agent_round"] < candidate["mean_tokens_per_agent_round"]
                 or other["mean_execution_calls_per_agent_round"] < candidate["mean_execution_calls_per_agent_round"])
            for other in grid
        )
        if not dominated:
            result.append(candidate["contract_id"])
    return sorted(result)


def select_contracts(grid: list[dict], presets: dict) -> dict:
    token_values = [x["mean_tokens_per_agent_round"] for x in grid]
    call_values = [x["mean_execution_calls_per_agent_round"] for x in grid]
    failure_values = [x["failure_rate"] for x in grid]
    selections = {}
    for name, weights in presets.items():
        scored = []
        for row in grid:
            utility = (
                row["accuracy"]
                - float(weights["token"]) * minmax(row["mean_tokens_per_agent_round"], token_values)
                - float(weights["call"]) * minmax(row["mean_execution_calls_per_agent_round"], call_values)
                - float(weights["failure"]) * minmax(row["failure_rate"], failure_values)
            )
            scored.append({**row, "leader_utility": utility})
        scored.sort(key=lambda x: (-x["leader_utility"],
                                   x["mean_execution_calls_per_agent_round"],
                                   x["mean_tokens_per_agent_round"], x["contract_id"]))
        selections[name] = {"selected": scored[0], "ranking": scored}
    return selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    records = load_jsonl(args.records)
    grid = summarize(records)
    expected = len(matrix["jobs"]) * int(matrix["defaults"]["n_samples"])
    observed = sum(x["n_questions"] for x in grid)
    if observed != expected:
        raise ValueError(f"grid incomplete: expected {expected} completed records, observed {observed}")
    report = {
        "claim_scope": "development-sample maximum over observed policies in the frozen finite grid; not a Stackelberg equilibrium certificate",
        "matrix_version": matrix["matrix_version"],
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "record_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.records},
        "normalization": "min-max within the complete development grid",
        "leader_utility": "accuracy - lambda_token*normalized_tokens - lambda_call*normalized_calls - lambda_failure*normalized_failure",
        "presets": matrix["leader_utility_presets"],
        "response_model": response_model(records),
        "grid": grid,
        "pareto_frontier": pareto_frontier(grid),
        "selections": select_contracts(grid, matrix["leader_utility_presets"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
