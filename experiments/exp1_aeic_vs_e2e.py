"""
Experiment 1: AEIC Layered Similarity vs End-to-End Similarity

论点: AEIC四层解耦评估具有更好的区分度和可解释性
Key insight: E2E inflates scores for partial disagreement.
  AEIC correctly identifies per-layer conflicts.
Metrics:
  - Discrimination ratio: gap between high/low agreement pairs
  - Per-layer conflict attribution (interpretability)
  - STS-B correlation (external validity)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from itertools import combinations
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from core.consensus import ConsensusEngine
from experiments.dataset_loader import load_reasoning_datasets, load_stsb_sample
from experiments.plot_config import (
    ieee_style, DOUBLE_COL, PALETTE, add_subfig_label
)

ieee_style()


def run(sim_method: str = "embedding"):
    print("=" * 60)
    print("Experiment 1: AEIC Layered vs End-to-End Similarity")
    print("=" * 60)

    # Load real reasoning datasets
    print("  Loading reasoning datasets...")
    all_tasks = load_reasoning_datasets(n_strategyqa=100, n_hotpotqa=100,
                                         agents_per_task=3, seed=42)

    # Split into high/low agreement based on source characteristics
    # StrategyQA tasks have higher structural agreement (shared fact base)
    # Create LOW agreement by cross-mixing nodes between unrelated tasks
    import random
    rng = random.Random(123)

    mid = len(all_tasks) // 2
    dataset_high = all_tasks[:mid]  # first half as-is (natural agreement)
    dataset_low = []
    for task in all_tasks[mid:]:
        # Swap one node with a node from a different task (break agreement)
        donor = rng.choice(all_tasks[:mid])
        new_nodes = list(task["nodes"])
        new_nodes[2] = donor["nodes"][0]  # replace 3rd agent with unrelated
        dataset_low.append({"nodes": new_nodes, **{k: v for k, v in task.items() if k != "nodes"}})

    engines = {
        "E2E": ConsensusEngine(similarity_method=sim_method),
        "AEIC-Uniform": ConsensusEngine(similarity_method=sim_method,
                                         weights={"A": 0.25, "E": 0.25, "I": 0.25, "C": 0.25}),
        "AEIC-Weighted": ConsensusEngine(similarity_method=sim_method,
                                          weights={"A": 0.15, "E": 0.35, "I": 0.20, "C": 0.30}),
    }

    results_high = {m: [] for m in engines}
    results_low = {m: [] for m in engines}
    layer_scores_high = {"A": [], "E": [], "I": [], "C": []}
    layer_scores_low = {"A": [], "E": [], "I": [], "C": []}

    for task in dataset_high:
        records = task["nodes"]
        # E2E
        e2e_sims = [engines["E2E"].end_to_end_similarity(records[i], records[j])
                    for i, j in combinations(range(len(records)), 2)]
        results_high["E2E"].append(np.mean(e2e_sims))
        # AEIC
        for name in ["AEIC-Uniform", "AEIC-Weighted"]:
            ev = engines[name].evaluate(records)
            results_high[name].append(ev["avg_similarity"])
            if name == "AEIC-Weighted":
                for k in layer_scores_high:
                    layer_scores_high[k].append(ev["dim_scores"][k])

    for task in dataset_low:
        records = task["nodes"]
        e2e_sims = [engines["E2E"].end_to_end_similarity(records[i], records[j])
                    for i, j in combinations(range(len(records)), 2)]
        results_low["E2E"].append(np.mean(e2e_sims))
        for name in ["AEIC-Uniform", "AEIC-Weighted"]:
            ev = engines[name].evaluate(records)
            results_low[name].append(ev["avg_similarity"])
            if name == "AEIC-Weighted":
                for k in layer_scores_low:
                    layer_scores_low[k].append(ev["dim_scores"][k])

    # Discrimination ratio: (high_mean - low_mean) / std
    disc_ratios = {}
    for m in engines:
        gap = np.mean(results_high[m]) - np.mean(results_low[m])
        pooled_std = np.sqrt((np.std(results_high[m])**2 + np.std(results_low[m])**2) / 2)
        disc_ratios[m] = gap / max(pooled_std, 1e-6)

    print(f"  {'Method':15s} {'High':>8s} {'Low':>8s} {'Gap':>8s} {'d-ratio':>8s}")
    for m in engines:
        print(f"  {m:15s} {np.mean(results_high[m]):>8.4f} {np.mean(results_low[m]):>8.4f} "
              f"{np.mean(results_high[m])-np.mean(results_low[m]):>8.4f} {disc_ratios[m]:>8.3f}")

    # STS-B validation (test similarity module quality)
    stsb = load_stsb_sample()
    from core.similarity import SimilarityCalculator
    calc = SimilarityCalculator(method=sim_method)
    pred_scores = [calc.compute(p["s1"], p["s2"]) for p in stsb]
    true_scores = [p["score"] for p in stsb]
    spearman_r, spearman_p = stats.spearmanr(pred_scores, true_scores)
    pearson_r, pearson_p = stats.pearsonr(pred_scores, true_scores)
    print(f"  STS-B Spearman: r={spearman_r:.4f}, p={spearman_p:.2e}")
    print(f"  STS-B Pearson:  r={pearson_r:.4f}, p={pearson_p:.2e}")

    # ============ IEEE Figure: 2x2 multi-panel ============
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.72))

    # (a) Discrimination comparison
    ax = axes[0, 0]
    methods = list(engines.keys())
    x = np.arange(len(methods))
    width = 0.3
    high_means = [np.mean(results_high[m]) for m in methods]
    low_means = [np.mean(results_low[m]) for m in methods]
    high_ci = [1.96 * np.std(results_high[m]) / np.sqrt(50) for m in methods]
    low_ci = [1.96 * np.std(results_low[m]) / np.sqrt(50) for m in methods]

    bars1 = ax.bar(x - width / 2, high_means, width, yerr=high_ci, capsize=2,
                   label="High Agreement", color=PALETTE[0], edgecolor="black", linewidth=0.4,
                   error_kw={"linewidth": 0.5})
    bars2 = ax.bar(x + width / 2, low_means, width, yerr=low_ci, capsize=2,
                   label="Low Agreement", color=PALETTE[1], edgecolor="black", linewidth=0.4,
                   error_kw={"linewidth": 0.5})
    # Annotate gap
    for i, m in enumerate(methods):
        gap = high_means[i] - low_means[i]
        mid_y = (high_means[i] + low_means[i]) / 2
        ax.annotate(f"Δ={gap:.3f}", xy=(i + 0.35, mid_y), fontsize=5.5, color=PALETTE[2])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel("Avg. Similarity")
    ax.set_title("Discrimination: High vs Low Agreement")
    ax.legend(fontsize=6, loc="upper left")
    add_subfig_label(ax, "a")

    # (b) Discrimination ratio (Cohen's d analog)
    ax = axes[0, 1]
    ratios = [disc_ratios[m] for m in methods]
    colors_b = [PALETTE[3], PALETTE[0], PALETTE[1]]
    bars = ax.bar(x, ratios, width=0.5, color=colors_b,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel("Discrimination Ratio (d')")
    ax.set_title("Separation Ability (Higher = Better)")
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{r:.2f}", ha="center", fontsize=7, fontweight="bold")
    add_subfig_label(ax, "b")

    # (c) Per-layer breakdown (high vs low)
    ax = axes[1, 0]
    layers = ["A", "E", "I", "C"]
    layer_labels = ["Assumption", "Evidence", "Inference", "Conclusion"]
    high_lmeans = [np.mean(layer_scores_high[l]) for l in layers]
    low_lmeans = [np.mean(layer_scores_low[l]) for l in layers]
    x_l = np.arange(4)
    ax.bar(x_l - 0.18, high_lmeans, 0.32, label="High Agr.",
           color=PALETTE[0], edgecolor="black", linewidth=0.4)
    ax.bar(x_l + 0.18, low_lmeans, 0.32, label="Low Agr.",
           color=PALETTE[1], edgecolor="black", linewidth=0.4)
    # Gap annotation
    for i in range(4):
        gap = high_lmeans[i] - low_lmeans[i]
        ax.annotate(f"Δ={gap:.2f}", xy=(i, max(high_lmeans[i], low_lmeans[i]) + 0.02),
                    fontsize=5.5, ha="center", color=PALETTE[2])
    ax.set_xticks(x_l)
    ax.set_xticklabels(layer_labels, fontsize=6.5)
    ax.set_ylabel("Layer Similarity")
    ax.set_title("AEIC Layer-wise Conflict Localization")
    ax.legend(fontsize=6)
    add_subfig_label(ax, "c")

    # (d) STS-B correlation scatter
    ax = axes[1, 1]
    ax.scatter(true_scores, pred_scores, s=8, alpha=0.6, color=PALETTE[0], edgecolors="none")
    # Fit line
    z = np.polyfit(true_scores, pred_scores, 1)
    p_line = np.poly1d(z)
    xs = np.linspace(0, 1, 100)
    ax.plot(xs, p_line(xs), "--", color=PALETTE[1], linewidth=0.8)
    ax.set_xlabel("Ground Truth (STS-B)")
    ax.set_ylabel(f"Predicted ({sim_method})")
    ax.set_title(f"STS-B Validation (ρ={spearman_r:.3f})")
    ax.text(0.05, 0.85, f"Spearman ρ={spearman_r:.3f}\nPearson r={pearson_r:.3f}",
            transform=ax.transAxes, fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    add_subfig_label(ax, "d")

    plt.savefig("figures/exp1_aeic_vs_e2e.pdf")
    plt.savefig("figures/exp1_aeic_vs_e2e.png", dpi=300)
    plt.close()
    print("  → figures/exp1_aeic_vs_e2e.{pdf,png}")

    return results_high, results_low, disc_ratios, (spearman_r, pearson_r)


if __name__ == "__main__":
    run()
