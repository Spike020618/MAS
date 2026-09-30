"""
Pareto前沿实验 — 不同λ下的accuracy vs IC trade-off

Usage:
  python experiments/run_pareto.py
  python experiments/run_pareto.py --train MMLU-Hard StrategyQA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.optimize_consensus import optimize
from experiments.eval_generalization import run_generalization

LAMBDAS = [0.0, 0.3, 0.5, 1.0, 2.0]


def main():
    parser = argparse.ArgumentParser(description="Run Pareto frontier experiments")
    parser.add_argument("--train", nargs="+", default=["MMLU-Hard", "StrategyQA"],
                        help="Training datasets")
    parser.add_argument("--lambdas", nargs="+", type=float, default=LAMBDAS,
                        help="Lambda values to sweep")
    args = parser.parse_args()

    signals_paths = [f"experiments/results/signals/{ds}.json" for ds in args.train]
    for sp in signals_paths:
        if not os.path.exists(sp):
            print(f"ERROR: {sp} not found")
            return

    train_name = "+".join(args.train)
    results = []

    for lam in args.lambdas:
        print(f"\n{'='*60}")
        print(f"  λ = {lam}")
        print(f"{'='*60}")
        t0 = time.time()

        out_path = f"experiments/results/configs/pareto_{train_name}_lam{lam}.json"
        cfg, info = optimize(signals_paths, lambda_ic=lam, out_path=out_path)

        elapsed = time.time() - t0
        print(f"\n  Time: {elapsed/60:.1f} min")

        entry = run_generalization(
            out_path,
            train_datasets=args.train,
            train_accuracy=info["train_accuracy"],
        )

        results.append({
            "lambda": lam,
            "train_accuracy": info["train_accuracy"],
            "train_ic": info["train_ic"],
            "avg_test_accuracy": entry["avg_test_accuracy"] if entry else 0,
            "config_path": out_path,
            "elapsed_min": round(elapsed / 60, 1),
        })

    # Summary table
    print(f"\n{'='*60}")
    print(f"  PARETO SUMMARY: train={train_name}")
    print(f"{'='*60}")
    print(f"{'λ':>6} {'Train-Acc':>10} {'Train-IC':>10} {'Test-Acc':>10} {'Time':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['lambda']:>6.1f} {r['train_accuracy']:>9.1%} "
              f"{r['train_ic']:>9.1%} {r['avg_test_accuracy']:>9.1%} "
              f"{r['elapsed_min']:>6.1f}m")

    pareto_path = f"experiments/results/pareto_{train_name}.json"
    with open(pareto_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {pareto_path}")


if __name__ == "__main__":
    main()
