"""
Cross-topology transfer analysis: How well does θ* from one topology work on another?

Generates a transfer matrix showing what happens when you apply each topology's
optimized θ* to the signals from every other topology.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.core.consensus_config import ConsensusConfig
from experiments.optimize_consensus import evaluate_config, compute_ic_metric
from collections import Counter

TOPOS = ["parallel", "sequential", "star", "debate", "hierarchical"]
DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]


def majvote_accuracy(questions):
    correct = 0
    total = 0
    for q in questions:
        if "signals" not in q:
            continue
        stances = q["signals"]["stances"]
        counts = Counter(s.lower() for s in stances if s.lower() != "neutral")
        if not counts:
            continue
        decision = counts.most_common(1)[0][0]
        if decision == q["ground_truth"]:
            correct += 1
        total += 1
    return correct / max(total, 1)


def load_questions(topo):
    """Load all questions for a topology (combined across datasets)."""
    all_q = []
    for ds in DATASETS:
        suffix = f"_{topo}" if topo != "parallel" else ""
        path = f"experiments/results/signals/{ds}{suffix}.json"
        with open(path) as f:
            data = json.load(f)
        all_q.extend(data["questions"])
    return all_q, data["meta"]["n_agents"]


def main():
    print("=" * 60)
    print("CROSS-TOPOLOGY TRANSFER ANALYSIS")
    print("=" * 60)

    configs = {}
    for topo in TOPOS:
        cv_path = f"experiments/results/configs/opt_{topo}_cv.json"
        path = f"experiments/results/configs/opt_{topo}.json"
        cfg_path = cv_path if os.path.exists(cv_path) else path
        configs[topo] = ConsensusConfig.load(cfg_path)

    print("\nTransfer Matrix: rows = source θ*, columns = target topology signals")
    print("Each cell = AEIC+SG accuracy when applying source θ* to target signals\n")

    transfer = np.zeros((len(TOPOS), len(TOPOS)))
    mv_acc = np.zeros(len(TOPOS))

    for j, target_topo in enumerate(TOPOS):
        questions, n_agents = load_questions(target_topo)
        mv_acc[j] = majvote_accuracy(questions) * 100

        for i, source_topo in enumerate(TOPOS):
            result = evaluate_config(questions, configs[source_topo], n_agents)
            transfer[i, j] = result["accuracy"] * 100

    header = f"{'Source θ*':>15s}"
    for topo in TOPOS:
        header += f" {topo:>12s}"
    print(header)
    print("-" * (15 + 13 * len(TOPOS)))

    for i, source_topo in enumerate(TOPOS):
        line = f"{source_topo:>15s}"
        for j in range(len(TOPOS)):
            diff = transfer[i, j] - mv_acc[j]
            marker = "*" if i == j else " "
            line += f" {transfer[i, j]:>10.1f}%{marker}"
        print(line)

    print()
    line = f"{'MajVote':>15s}"
    for j in range(len(TOPOS)):
        line += f" {mv_acc[j]:>10.1f}% "
    print(line)

    print("\n\nDelta (AEIC+SG - MajVote):")
    header = f"{'Source θ*':>15s}"
    for topo in TOPOS:
        header += f" {topo:>12s}"
    print(header)
    print("-" * (15 + 13 * len(TOPOS)))

    for i, source_topo in enumerate(TOPOS):
        line = f"{source_topo:>15s}"
        for j in range(len(TOPOS)):
            diff = transfer[i, j] - mv_acc[j]
            marker = "*" if i == j else " "
            line += f" {diff:>+10.1f}pp{marker}"
        print(line)

    print("\n* = matched (θ* applied to its own topology)")

    best_transfer = np.max(transfer, axis=0)
    print(f"\nBest achievable with any θ*: {[f'{v:.1f}%' for v in best_transfer]}")
    print(f"MajVote:                     {[f'{v:.1f}%' for v in mv_acc]}")

    results = {
        "transfer_matrix": transfer.tolist(),
        "majvote_acc": mv_acc.tolist(),
        "topos": TOPOS,
    }
    out_path = "experiments/results/cross_topology_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
