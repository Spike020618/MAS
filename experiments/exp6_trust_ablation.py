"""
Experiment 6: Trust Component Ablation Study

论点: AEIC 信任机制的每个组件都不可或缺
Configurations:
  - Full: IC + EC + EvCons + Cal (complete model)
  - −IC: remove Inference→Conclusion consistency
  - −EC: remove Evidence→Conclusion entailment
  - −EvCons: remove cross-agent evidence consensus
  - −Cal: remove confidence calibration
  - Random: uniform random trust (no structure)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from experiments.baselines import OurMethod, _check_accuracy, _extract_stance
from experiments.dataset_loader import load_reasoning_datasets
from experiments.exp5_baseline_comparison import _create_adversarial_tasks
from experiments.plot_config import (
    ieee_style, DOUBLE_COL, PALETTE, add_subfig_label
)

ieee_style()

ABLATION_CONFIGS = {
    "Full": {"ic": 0.35, "ec": 0.20, "ev_cons": 0.25, "cal": 0.20},
    "−IC": {"ic": 0.00, "ec": 0.25, "ev_cons": 0.40, "cal": 0.35},
    "−EC": {"ic": 0.45, "ec": 0.00, "ev_cons": 0.30, "cal": 0.25},
    "−EvCons": {"ic": 0.45, "ec": 0.25, "ev_cons": 0.00, "cal": 0.30},
    "−Cal": {"ic": 0.45, "ec": 0.25, "ev_cons": 0.30, "cal": 0.00},
}


def run(sim_method: str = "embedding"):
    print("\n" + "=" * 60)
    print("Experiment 6: Trust Component Ablation Study")
    print("=" * 60)

    print("  Loading reasoning datasets...")
    dataset = load_reasoning_datasets(n_strategyqa=100, n_hotpotqa=100,
                                       agents_per_task=3, seed=42)
    boolean_tasks = [t for t in dataset if isinstance(t.get("ground_truth"), bool)]
    mid = len(boolean_tasks) // 2
    easy_tasks = boolean_tasks[:mid]
    hard_tasks = _create_adversarial_tasks(boolean_tasks[mid:],
                                           n_adversarial=len(boolean_tasks) - mid, seed=42)
    all_tasks = easy_tasks + hard_tasks
    print(f"  Tasks: {len(easy_tasks)} easy + {len(hard_tasks)} hard")

    results = {}
    for config_name, tw in ABLATION_CONFIGS.items():
        method = OurMethod(sim_method=sim_method, trust_weights=tw)
        easy_acc, hard_acc, all_acc = [], [], []

        for task in easy_tasks:
            r = method.consensus(task["nodes"], task.get("ground_truth"))
            acc = 1.0 if r["accuracy"] else 0.0
            easy_acc.append(acc)
            all_acc.append(acc)

        for task in hard_tasks:
            r = method.consensus(task["nodes"], task.get("ground_truth"))
            acc = 1.0 if r["accuracy"] else 0.0
            hard_acc.append(acc)
            all_acc.append(acc)

        results[config_name] = {
            "easy": np.mean(easy_acc), "hard": np.mean(hard_acc),
            "all": np.mean(all_acc),
            "easy_ci": 1.96 * np.std(easy_acc) / np.sqrt(max(len(easy_acc), 1)),
            "hard_ci": 1.96 * np.std(hard_acc) / np.sqrt(max(len(hard_acc), 1)),
            "hard_raw": hard_acc, "all_raw": all_acc,
        }
        print(f"  {config_name:10s}: easy={np.mean(easy_acc):.3f}  "
              f"hard={np.mean(hard_acc):.3f}  all={np.mean(all_acc):.3f}")

    # Statistical tests: Full vs each ablation
    p_values = {}
    for name in ABLATION_CONFIGS:
        if name == "Full":
            continue
        min_len = min(len(results["Full"]["all_raw"]), len(results[name]["all_raw"]))
        if min_len > 1:
            t, p = stats.ttest_rel(results["Full"]["all_raw"][:min_len],
                                   results[name]["all_raw"][:min_len])
            p_values[name] = p
            print(f"  t-test Full vs {name}: p={p:.2e}")

    # ============ IEEE Figure: 2x2 multi-panel ============
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))
    config_names = list(ABLATION_CONFIGS.keys())
    n_configs = len(config_names)
    x = np.arange(n_configs)
    colors = [PALETTE[3]] + [PALETTE[0]] * (n_configs - 1)

    # (a) Accuracy by difficulty
    ax = axes[0, 0]
    width = 0.35
    easy_vals = [results[c]["easy"] for c in config_names]
    hard_vals = [results[c]["hard"] for c in config_names]
    ax.bar(x - width / 2, easy_vals, width, label="Easy", color=PALETTE[2],
           edgecolor="black", linewidth=0.4, alpha=0.6)
    bars_h = ax.bar(x + width / 2, hard_vals, width, label="Hard", color=PALETTE[1],
                    edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars_h, hard_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", fontsize=5.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=6.5, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Task Difficulty")
    ax.legend(fontsize=6)
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "a")

    # (b) Overall accuracy with significance
    ax = axes[0, 1]
    all_vals = [results[c]["all"] for c in config_names]
    bars = ax.bar(x, all_vals, width=0.5, color=colors,
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, all_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=6, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=6.5, rotation=15)
    ax.set_ylabel("Overall Accuracy")
    ax.set_title("Ablation: Overall Performance")
    ax.set_ylim(0, max(all_vals) * 1.2)
    add_subfig_label(ax, "b")

    # (c) Accuracy drop from Full (delta)
    ax = axes[1, 0]
    full_hard = results["Full"]["hard"]
    drops = [full_hard - results[c]["hard"] for c in config_names]
    bar_colors = ["gray"] + [PALETTE[1] if d > 0 else PALETTE[2] for d in drops[1:]]
    bars = ax.bar(x, drops, width=0.5, color=bar_colors,
                  edgecolor="black", linewidth=0.5)
    for bar, d in zip(bars, drops):
        if abs(d) > 0.001:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * (1 if d >= 0 else -1),
                    f"{d:+.3f}", ha="center", fontsize=6,
                    va="bottom" if d >= 0 else "top")
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=6.5, rotation=15)
    ax.set_ylabel("Δ Hard Accuracy (vs Full)")
    ax.set_title("Component Contribution (Hard Tasks)")
    ax.axhline(0, color="black", linewidth=0.5)
    add_subfig_label(ax, "c")

    # (d) Trust weight distribution visualization
    ax = axes[1, 1]
    components = ["IC", "EC", "EvCons", "Cal"]
    comp_colors = ["#e08214", "#2166ac", "#1b7837", "#762a83"]
    bottom = np.zeros(n_configs)
    for i, comp_key in enumerate(["ic", "ec", "ev_cons", "cal"]):
        vals = [ABLATION_CONFIGS[c][comp_key] for c in config_names]
        ax.bar(x, vals, 0.5, bottom=bottom, label=components[i],
               color=comp_colors[i], edgecolor="black", linewidth=0.3)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=6.5, rotation=15)
    ax.set_ylabel("Weight Allocation")
    ax.set_title("Trust Weight Configurations")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "d")

    plt.savefig("figures/exp6_trust_ablation.pdf")
    plt.savefig("figures/exp6_trust_ablation.png", dpi=300)
    plt.close()
    print("  → figures/exp6_trust_ablation.{pdf,png}")

    return results, p_values


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default="embedding", choices=["embedding", "bm25", "tfidf"])
    args = parser.parse_args()
    run(sim_method=args.sim)
