"""
Experiment 2: Adaptive Weight Learning vs Fixed Weight Baselines

论点: 自适应权重学习（logits-softmax）优于所有固定权重方案
Methods:
  1. Uniform: [0.25, 0.25, 0.25, 0.25] (ChatEval-style)
  2. Evidence-Heavy: [0.10, 0.40, 0.10, 0.40]
  3. Structured: [0.15, 0.35, 0.15, 0.35] (LeaderFollowing-style)
  4. Adaptive (Ours): logits-softmax with feedback-driven update
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from core.consensus import ConsensusEngine
from experiments.dataset_loader import load_reasoning_datasets
from experiments.plot_config import (
    ieee_style, DOUBLE_COL, PALETTE, HATCHES,
    significance_marker, add_subfig_label
)

ieee_style()


def run(sim_method: str = "embedding"):
    print("\n" + "=" * 60)
    print("Experiment 2: Adaptive vs Fixed Weight Strategies")
    print("=" * 60)

    print("  Loading reasoning datasets...")
    dataset = load_reasoning_datasets(n_strategyqa=100, n_hotpotqa=100,
                                       agents_per_task=3, seed=42)

    methods_cfg = {
        "Uniform": {"A": 0.25, "E": 0.25, "I": 0.25, "C": 0.25},
        "Evidence-Heavy": {"A": 0.10, "E": 0.40, "I": 0.10, "C": 0.40},
        "Structured": {"A": 0.15, "E": 0.35, "I": 0.15, "C": 0.35},
    }

    results = {m: [] for m in list(methods_cfg.keys()) + ["Adaptive (Ours)"]}
    utility_results = {m: [] for m in results}

    # Fixed baselines
    for name, weights in methods_cfg.items():
        engine = ConsensusEngine(similarity_method=sim_method, weights=weights)
        for task in dataset:
            ev = engine.evaluate(task["nodes"])
            results[name].append(ev["avg_similarity"])
            utility_results[name].append(ev["utility"])

    # Adaptive method
    engine_ada = ConsensusEngine(
        similarity_method=sim_method,
        weights={"A": 0.25, "E": 0.25, "I": 0.25, "C": 0.25},
    )
    weight_history = [dict(engine_ada.w)]
    utility_curve = []

    num_rounds = 30
    for rnd in range(num_rounds):
        round_utility = []
        for task in dataset:
            ev = engine_ada.evaluate(task["nodes"])
            engine_ada.update_weights(ev["utility"], lr=0.03, dim_scores=ev["dim_scores"])
            round_utility.append(ev["utility"])
        weight_history.append(dict(engine_ada.w))
        utility_curve.append(np.mean(round_utility))

    # Final evaluation with learned weights
    for task in dataset:
        ev = engine_ada.evaluate(task["nodes"])
        results["Adaptive (Ours)"].append(ev["avg_similarity"])
        utility_results["Adaptive (Ours)"].append(ev["utility"])

    # Print
    for m in results:
        print(f"  {m:18s}: sim={np.mean(results[m]):.4f}±{np.std(results[m]):.4f}  "
              f"utility={np.mean(utility_results[m]):.2f}")

    # Statistical tests
    t_stat, p_ada_uni = stats.ttest_rel(results["Adaptive (Ours)"], results["Uniform"])
    print(f"  t-test Adaptive vs Uniform: t={t_stat:.3f}, p={p_ada_uni:.2e}")

    # ============ IEEE Figure: 2x2 multi-panel ============
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.72))
    method_names = list(results.keys())

    # (a) Method comparison - similarity
    ax = axes[0, 0]
    means = [np.mean(results[m]) for m in method_names]
    ci95 = [1.96 * np.std(results[m]) / np.sqrt(len(results[m])) for m in method_names]
    x = np.arange(len(method_names))
    colors = PALETTE[:4]
    bars = ax.bar(x, means, yerr=ci95, capsize=3, width=0.6,
                  color=colors, edgecolor="black", linewidth=0.5,
                  error_kw={"linewidth": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(["Unif.", "Evid.", "Struct.", "Ours"], fontsize=7)
    ax.set_ylabel("Avg. Consensus Similarity")
    ax.set_title("Method Comparison (Similarity)")
    ax.set_ylim(0, max(means) * 1.3)
    significance_marker(ax, 0, 3, max(means) + max(ci95) + 0.005, p_ada_uni)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{m:.3f}", ha="center", fontsize=6)
    add_subfig_label(ax, "a")

    # (b) Method comparison - utility
    ax = axes[0, 1]
    u_means = [np.mean(utility_results[m]) for m in method_names]
    u_ci = [1.96 * np.std(utility_results[m]) / np.sqrt(len(utility_results[m]))
            for m in method_names]
    bars = ax.bar(x, u_means, yerr=u_ci, capsize=3, width=0.6,
                  color=colors, edgecolor="black", linewidth=0.5,
                  error_kw={"linewidth": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(["Unif.", "Evid.", "Struct.", "Ours"], fontsize=7)
    ax.set_ylabel("Utility Score")
    ax.set_title("Method Comparison (Utility)")
    for bar, m in zip(bars, u_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{m:.1f}", ha="center", fontsize=6)
    add_subfig_label(ax, "b")

    # (c) Weight convergence over rounds
    ax = axes[1, 0]
    rounds = np.arange(len(weight_history))
    layer_colors = {"A": "#f4a582", "E": "#2166ac", "I": "#1b7837", "C": "#762a83"}
    markers = {"A": "o", "E": "s", "I": "^", "C": "D"}
    for key in ["A", "E", "I", "C"]:
        vals = [wh[key] for wh in weight_history]
        ax.plot(rounds, vals, f"-{markers[key]}", color=layer_colors[key],
                linewidth=1.0, markersize=3, markevery=3,
                label=f"$w_{{{key}}}$")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Weight Value")
    ax.set_title("Adaptive Weight Convergence")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    add_subfig_label(ax, "c")

    # (d) Utility improvement over rounds
    ax = axes[1, 1]
    ax.plot(range(1, num_rounds + 1), utility_curve, "o-",
            color=PALETTE[3], linewidth=1.0, markersize=3, markevery=3)
    ax.axhline(np.mean(utility_results["Uniform"]), color=PALETTE[0],
               linestyle="--", linewidth=0.8, label="Uniform baseline")
    ax.axhline(np.mean(utility_results["Structured"]), color=PALETTE[2],
               linestyle="-.", linewidth=0.8, label="Structured baseline")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Average Utility")
    ax.set_title("Utility Curve During Adaptation")
    ax.legend(fontsize=6, loc="lower right")
    add_subfig_label(ax, "d")

    plt.savefig("figures/exp2_adaptive_weights.pdf")
    plt.savefig("figures/exp2_adaptive_weights.png", dpi=300)
    plt.close()
    print("  → figures/exp2_adaptive_weights.{pdf,png}")

    return results, weight_history, utility_curve


if __name__ == "__main__":
    run()
