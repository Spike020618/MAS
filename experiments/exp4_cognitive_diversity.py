"""
Experiment 4: Cognitive Diversity Ablation Study

论点: 异构多Agent通过认知多样性提升信息覆盖和鲁棒性
Comparison:
  - Homogeneous: 3 agents with identical perspectives
  - Heterogeneous: 3 agents with different disciplinary perspectives
Metrics:
  - Consensus similarity (internal agreement)
  - Information coverage (evidence diversity)
  - Conflict attribution distribution
  - Per-layer similarity breakdown
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from core.consensus import ConsensusEngine
from experiments.dataset_loader import generate_diversity_dataset
from experiments.plot_config import (
    ieee_style, DOUBLE_COL, PALETTE, HATCHES, add_subfig_label
)

ieee_style()


def information_coverage(nodes):
    """Measure evidence diversity: unique evidence items / total items."""
    all_evidence = []
    unique_evidence = set()
    for node in nodes:
        evs = node.get("E", [])
        if isinstance(evs, list):
            all_evidence.extend(evs)
            unique_evidence.update(evs)
        else:
            all_evidence.append(evs)
            unique_evidence.add(evs)
    if not all_evidence:
        return 0.0
    return len(unique_evidence) / len(all_evidence)


def run(sim_method: str = "embedding"):
    print("\n" + "=" * 60)
    print("Experiment 4: Cognitive Diversity Ablation")
    print("=" * 60)

    data = generate_diversity_dataset(seed=42)
    engine = ConsensusEngine(similarity_method=sim_method,
                             weights={"A": 0.15, "E": 0.35, "I": 0.20, "C": 0.30})

    homo_sims, hetero_sims = [], []
    homo_utility, hetero_utility = [], []
    homo_coverage, hetero_coverage = [], []
    homo_layers = {"A": [], "E": [], "I": [], "C": []}
    hetero_layers = {"A": [], "E": [], "I": [], "C": []}
    homo_conflicts = {"A": 0, "E": 0, "I": 0, "C": 0}
    hetero_conflicts = {"A": 0, "E": 0, "I": 0, "C": 0}
    topics = []

    for task in data["homogeneous"]:
        topics.append(task["topic"][:12])
        ev = engine.evaluate(task["nodes"])
        homo_sims.append(ev["avg_similarity"])
        homo_utility.append(ev["utility"])
        homo_coverage.append(information_coverage(task["nodes"]))
        for k in homo_layers:
            homo_layers[k].append(ev["dim_scores"][k])
        for layer in ev["conflict_attribution"].values():
            homo_conflicts[layer] += 1

    for task in data["heterogeneous"]:
        ev = engine.evaluate(task["nodes"])
        hetero_sims.append(ev["avg_similarity"])
        hetero_utility.append(ev["utility"])
        hetero_coverage.append(information_coverage(task["nodes"]))
        for k in hetero_layers:
            hetero_layers[k].append(ev["dim_scores"][k])
        for layer in ev["conflict_attribution"].values():
            hetero_conflicts[layer] += 1

    print(f"  Homogeneous:  sim={np.mean(homo_sims):.4f}±{np.std(homo_sims):.4f}  "
          f"coverage={np.mean(homo_coverage):.3f}")
    print(f"  Heterogeneous: sim={np.mean(hetero_sims):.4f}±{np.std(hetero_sims):.4f}  "
          f"coverage={np.mean(hetero_coverage):.3f}")

    t_stat, p_value = stats.ttest_ind(homo_sims, hetero_sims)
    print(f"  t-test: t={t_stat:.3f}, p={p_value:.2e}")

    # ============ IEEE Figure: 2x2 multi-panel ============
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.72))

    # (a) Grouped bar: similarity per topic (first 10)
    ax = axes[0, 0]
    n_show = min(10, len(homo_sims))
    x = np.arange(n_show)
    width = 0.35
    ax.bar(x - width / 2, homo_sims[:n_show], width, label="Homogeneous",
           color=PALETTE[0], edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, hetero_sims[:n_show], width, label="Heterogeneous",
           color=PALETTE[1], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([t[:6] for t in topics[:n_show]], fontsize=5.5, rotation=30, ha="right")
    ax.set_ylabel("Consensus Similarity")
    ax.set_title("Per-Topic Comparison")
    ax.legend(fontsize=6)
    ax.axhline(np.mean(homo_sims), color=PALETTE[0], linestyle="--", linewidth=0.5, alpha=0.6)
    ax.axhline(np.mean(hetero_sims), color=PALETTE[1], linestyle="--", linewidth=0.5, alpha=0.6)
    add_subfig_label(ax, "a")

    # (b) Radar-style: layer breakdown comparison
    ax = axes[0, 1]
    layers = ["A", "E", "I", "C"]
    homo_means = [np.mean(homo_layers[l]) for l in layers]
    hetero_means = [np.mean(hetero_layers[l]) for l in layers]
    x_l = np.arange(len(layers))
    width_l = 0.35
    ax.bar(x_l - width_l / 2, homo_means, width_l, label="Homo",
           color=PALETTE[0], edgecolor="black", linewidth=0.4, alpha=0.8)
    ax.bar(x_l + width_l / 2, hetero_means, width_l, label="Hetero",
           color=PALETTE[1], edgecolor="black", linewidth=0.4, alpha=0.8)
    ax.set_xticks(x_l)
    ax.set_xticklabels(["Assumption", "Evidence", "Inference", "Conclusion"], fontsize=6.5)
    ax.set_ylabel("Layer Similarity")
    ax.set_title("Per-Layer Similarity Breakdown")
    ax.legend(fontsize=6)
    add_subfig_label(ax, "b")

    # (c) Information coverage comparison + utility
    ax = axes[1, 0]
    categories = ["Similarity", "Coverage", "Utility (norm)"]
    homo_vals = [np.mean(homo_sims), np.mean(homo_coverage),
                 np.mean(homo_utility) / max(np.mean(homo_utility), np.mean(hetero_utility), 1)]
    hetero_vals = [np.mean(hetero_sims), np.mean(hetero_coverage),
                   np.mean(hetero_utility) / max(np.mean(homo_utility), np.mean(hetero_utility), 1)]
    x_c = np.arange(len(categories))
    ax.bar(x_c - 0.2, homo_vals, 0.35, label="Homogeneous",
           color=PALETTE[0], edgecolor="black", linewidth=0.4)
    ax.bar(x_c + 0.2, hetero_vals, 0.35, label="Heterogeneous",
           color=PALETTE[1], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x_c)
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("Multi-Metric Comparison")
    ax.legend(fontsize=6)
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "c")

    # (d) Conflict attribution stacked bar
    ax = axes[1, 1]
    homo_total = max(sum(homo_conflicts.values()), 1)
    hetero_total = max(sum(hetero_conflicts.values()), 1)
    layer_colors = ["#f4a582", "#92c5de", "#a6d96a", "#d5a6bd"]

    categories_ca = ["Homogeneous", "Heterogeneous"]
    bottom = [0, 0]
    for i, layer in enumerate(layers):
        vals = [homo_conflicts[layer] / homo_total, hetero_conflicts[layer] / hetero_total]
        ax.bar([0, 1], vals, 0.5, bottom=bottom, label=f"Layer {layer}",
               color=layer_colors[i], edgecolor="black", linewidth=0.4)
        # Add percentage text
        for j in range(2):
            if vals[j] > 0.05:
                ax.text(j, bottom[j] + vals[j] / 2, f"{vals[j]*100:.0f}%",
                        ha="center", va="center", fontsize=6)
        bottom = [bottom[k] + vals[k] for k in range(2)]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(categories_ca, fontsize=7)
    ax.set_ylabel("Conflict Attribution Ratio")
    ax.set_title("Conflict Source Distribution")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "d")

    plt.savefig("figures/exp4_cognitive_diversity.pdf")
    plt.savefig("figures/exp4_cognitive_diversity.png", dpi=300)
    plt.close()
    print("  → figures/exp4_cognitive_diversity.{pdf,png}")

    return {
        "homo_sims": homo_sims, "hetero_sims": hetero_sims,
        "homo_coverage": homo_coverage, "hetero_coverage": hetero_coverage,
        "homo_conflicts": homo_conflicts, "hetero_conflicts": hetero_conflicts,
    }


if __name__ == "__main__":
    run()
