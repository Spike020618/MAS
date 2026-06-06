"""
Fast Pareto sweep — smaller population for quicker results.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import (
    evaluate_config, compute_ic_metric, objective,
)
from experiments.eval_generalization import evaluate_on_all, find_all_signals, extract_single_agent_baseline

SIGNALS_PATHS = [
    "results/signals/MMLU-Hard.json",
    "results/signals/StrategyQA.json",
]
TRAIN_DS = ["MMLU-Hard", "StrategyQA"]
LAMBDAS = [0.0, 0.3, 0.5, 1.0, 2.0]


def optimize_fast(signals_paths, lambda_ic, maxiter=40, popsize=8):
    from scipy.optimize import differential_evolution

    all_questions = []
    n_agents = None
    for sp in signals_paths:
        with open(sp, "r") as f:
            data = json.load(f)
        all_questions.extend(data["questions"])
        n_agents = data["meta"]["n_agents"]

    bounds = ConsensusConfig.bounds()
    result = differential_evolution(
        objective, bounds,
        args=(all_questions, n_agents, lambda_ic),
        seed=42, maxiter=maxiter, popsize=popsize,
        tol=1e-4, polish=True, init="latinhypercube",
        workers=1, disp=False,
    )

    best_cfg = ConsensusConfig.from_vector(result.x)
    train_result = evaluate_config(all_questions, best_cfg, n_agents)
    train_ic = compute_ic_metric(all_questions, best_cfg, n_agents)
    return best_cfg, train_result["accuracy"], train_ic


def main():
    signal_files = find_all_signals("results")
    results = []

    for lam in LAMBDAS:
        print(f"Optimizing λ={lam}...", end=" ", flush=True)
        cfg, train_acc, train_ic = optimize_fast(SIGNALS_PATHS, lam)

        test_accs = []
        all_results = evaluate_on_all(cfg, signal_files)
        for ds, res in all_results.items():
            if ds not in TRAIN_DS:
                test_accs.append(res["accuracy"])

        test_avg = sum(test_accs) / len(test_accs) if test_accs else 0

        results.append({
            "lambda": lam,
            "train_accuracy": train_acc,
            "train_ic": train_ic,
            "test_accuracy": test_avg,
        })
        print(f"train_acc={train_acc:.1%} train_ic={train_ic:.1%} test_acc={test_avg:.1%}")

    print(f"\n{'λ':>6} {'Train-Acc':>10} {'Train-IC':>10} {'Test-Acc':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['lambda']:>6.1f} {r['train_accuracy']:>9.1%} "
              f"{r['train_ic']:>9.1%} {r['test_accuracy']:>9.1%}")

    with open("results/pareto_fast.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/pareto_fast.json")


if __name__ == "__main__":
    main()
