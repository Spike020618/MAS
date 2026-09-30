"""
Per-Topology θ* Optimization — 每种拓扑独立训练参数

对每种拓扑加载其3个数据集的signals，联合优化θ*。
零API开销，纯本地DE计算。

Usage:
  python experiments/optimize_per_topology.py
  python experiments/optimize_per_topology.py --topology sequential
  python experiments/optimize_per_topology.py --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config, compute_ic_metric, objective

TOPOLOGIES = ["parallel", "sequential", "star", "debate", "agentverse"]
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
SIGNALS_DIR = "experiments/results/signals"
CONFIGS_DIR = "experiments/results/configs"


def load_topology_signals(topology: str, datasets: list = None,
                          max_per_dataset: int = None):
    """加载指定拓扑的所有数据集signals。"""
    datasets = datasets or DATASETS
    all_questions = []
    n_agents = None

    for ds in datasets:
        if topology == "parallel":
            path = os.path.join(SIGNALS_DIR, f"{ds}.json")
        else:
            path = os.path.join(SIGNALS_DIR, f"{ds}_{topology}.json")

        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data["questions"]
        questions = [q for q in questions if "signals" in q]
        if max_per_dataset and len(questions) > max_per_dataset:
            questions = questions[:max_per_dataset]
        all_questions.extend(questions)
        n_agents = data["meta"]["n_agents"]

    return all_questions, n_agents


def optimize_topology(topology: str, lambda_ic: float = 0.3,
                      maxiter: int = 100, popsize: int = 10,
                      max_per_dataset: int = None):
    """对指定拓扑运行DE优化。"""
    from scipy.optimize import differential_evolution

    all_questions, n_agents = load_topology_signals(topology,
                                                    max_per_dataset=max_per_dataset)
    if not all_questions:
        print(f"  ERROR: No signals data for topology '{topology}'")
        return None, None

    print(f"  Loaded {len(all_questions)} questions, {n_agents} agents")

    bounds = ConsensusConfig.bounds()
    result = differential_evolution(
        objective, bounds,
        args=(all_questions, n_agents, lambda_ic),
        seed=42, maxiter=maxiter, popsize=popsize,
        tol=1e-3, polish=False, init="latinhypercube",
        workers=1, disp=False,
    )

    best_cfg = ConsensusConfig.from_vector(result.x)
    train_result = evaluate_config(all_questions, best_cfg, n_agents)
    train_ic = compute_ic_metric(all_questions, best_cfg, n_agents)

    return best_cfg, {
        "topology": topology,
        "accuracy": train_result["accuracy"],
        "ic_metric": train_ic,
        "n_questions": len(all_questions),
        "converged": result.success,
        "n_iterations": result.nit,
    }


def verify():
    """验证模式：检查signals文件是否存在。"""
    print("=== Verification: Checking signals availability ===\n")
    all_ok = True

    for topo in TOPOLOGIES:
        print(f"Topology: {topo}")
        for ds in DATASETS:
            if topo == "parallel":
                path = os.path.join(SIGNALS_DIR, f"{ds}.json")
            else:
                path = os.path.join(SIGNALS_DIR, f"{ds}_{topo}.json")

            exists = os.path.exists(path)
            if exists:
                with open(path, "r") as f:
                    data = json.load(f)
                n_q = len(data["questions"])
                has_signals = sum(1 for q in data["questions"] if "signals" in q)
                print(f"  {ds}: {path} ({n_q} questions, {has_signals} with signals)")
            else:
                print(f"  {ds}: MISSING {path}")
                all_ok = False
        print()

    if all_ok:
        print("All signals files present. Ready to optimize.")
    else:
        print("Some files missing. Run collect_cache_topo.py first.")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Per-Topology Optimization")
    parser.add_argument("--topology", default="all",
                        choices=TOPOLOGIES + ["all"])
    parser.add_argument("--lambda_ic", type=float, default=0.3)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--max_per_dataset", type=int, default=None,
                        help="Max questions per dataset for training")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    os.makedirs(CONFIGS_DIR, exist_ok=True)

    if args.verify:
        verify()
        return

    topos = TOPOLOGIES if args.topology == "all" else [args.topology]

    results_summary = {}
    for topo in topos:
        print(f"\n{'='*50}")
        print(f"Optimizing θ* for topology: {topo}")
        print(f"{'='*50}")

        cfg, info = optimize_topology(
            topo, lambda_ic=args.lambda_ic,
            maxiter=args.maxiter, popsize=args.popsize,
            max_per_dataset=args.max_per_dataset)

        if cfg is None:
            continue

        out_path = os.path.join(CONFIGS_DIR, f"opt_{topo}.json")
        cfg.save(out_path)
        print(f"  Accuracy: {info['accuracy']:.1%}")
        print(f"  IC metric: {info['ic_metric']:.1%}")
        print(f"  Saved to {out_path}")

        results_summary[topo] = info

    # 汇总表
    if results_summary:
        print(f"\n{'='*50}")
        print(f"{'Topology':<14} {'Accuracy':>10} {'IC':>10} {'Questions':>10}")
        print("-" * 50)
        for topo, info in results_summary.items():
            print(f"{topo:<14} {info['accuracy']:>9.1%} "
                  f"{info['ic_metric']:>9.1%} {info['n_questions']:>10}")

        summary_path = os.path.join(CONFIGS_DIR, "optimization_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
