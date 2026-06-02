"""
Experiment 9: Cross-Dataset Generalization

论点: AEIC-Stackelberg 框架在不同数据集、语言、难度、领域上均保持优势。

Datasets (8 total):
  - StrategyQA (EN, medium, boolean reasoning)
  - HotpotQA (EN, medium, multi-hop QA)
  - GPQA (EN, medium, graduate science MCQ)
  - GAIA (EN, hard, multi-modal daily tasks)
  - BrowseComp (EN, hard, browsing comprehension)
  - BrowseComp-ZH (ZH, hard, Chinese browsing comprehension)
  - HLE (EN, hard, extreme multi-discipline)
  - ScholarSearch (EN, medium, academic retrieval)

Comparison: Ours vs ChatEval vs Unguided Debate vs AgentVerse
Metrics: Accuracy, Oracle Gap, Info Utilization, Consensus Strength
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from experiments.baselines import (
    ChatEvalBaseline, UnguidedDebateBaseline, AgentVerseBaseline, OurMethod
)
from experiments.dataset_loader import load_single_dataset, DATASET_REGISTRY
from experiments.exp5_baseline_comparison import _create_adversarial_tasks
from experiments.plot_config import (
    ieee_style, DOUBLE_COL, PALETTE, add_subfig_label
)

ieee_style()

METHOD_NAMES = ["ChatEval", "Unguided Debate", "AgentVerse", "Ours"]
METHOD_COLORS = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
METHOD_SHORT = ["CE", "UD", "AV", "Ours"]

DATASETS_TO_TEST = list(DATASET_REGISTRY.keys())
DATASET_SHORT = {
    "StrategyQA": "SQA",
    "HotpotQA": "HQA",
    "GPQA": "GPQA",
    "GAIA": "GAIA",
    "BrowseComp": "BC",
    "BrowseComp-ZH": "BC-ZH",
    "HLE": "HLE",
    "ScholarSearch": "SS",
}


def _evaluate_methods(tasks, sim_method, seed=42):
    """Run all 4 methods on a set of tasks."""
    chateval = ChatEvalBaseline(sim_method=sim_method)
    debate = UnguidedDebateBaseline(sim_method=sim_method, seed=seed)
    agentverse = AgentVerseBaseline(sim_method=sim_method)
    ours = OurMethod(sim_method=sim_method)

    results = {m: [] for m in METHOD_NAMES}

    for task in tasks:
        gt = task.get("ground_truth")
        nodes = task["nodes"]

        results["ChatEval"].append(chateval.consensus(nodes, gt))
        results["Unguided Debate"].append(debate.consensus(nodes, gt))
        results["AgentVerse"].append(agentverse.consensus(nodes, gt))
        results["Ours"].append(ours.consensus(nodes, gt))

    return results


def _compute_metrics(results):
    """Compute aggregate metrics from raw results."""
    metrics = {}
    for method, res_list in results.items():
        accs = [1.0 if r["accuracy"] else 0.0 for r in res_list if r["accuracy"] is not None]
        metrics[method] = {
            "accuracy": np.mean(accs) if accs else 0.0,
            "accuracy_ci": 1.96 * np.std(accs) / np.sqrt(max(len(accs), 1)) if accs else 0.0,
            "oracle_gap": np.mean([r["oracle_gap"] for r in res_list]),
            "info_utilization": np.mean([r["info_utilization"] for r in res_list]),
            "consensus_strength": np.mean([r["consensus_strength"] for r in res_list]),
            "similarity": np.mean([r["similarity"] for r in res_list]),
            "n_tasks": len(res_list),
        }
    return metrics


def run(sim_method: str = "embedding", n_per_dataset: int = 50):
    print("\n" + "=" * 60)
    print("Experiment 9: Cross-Dataset Generalization")
    print("=" * 60)

    all_dataset_results = {}
    all_dataset_metrics = {}
    loaded_datasets = []

    for ds_name in DATASETS_TO_TEST:
        tasks = load_single_dataset(ds_name, n_samples=n_per_dataset,
                                    agents_per_task=3, seed=42)
        if not tasks:
            print(f"  ⚠ Skipping {ds_name} (failed to load)")
            continue

        loaded_datasets.append(ds_name)
        print(f"  Evaluating {ds_name} ({len(tasks)} tasks)...")
        results = _evaluate_methods(tasks, sim_method)
        metrics = _compute_metrics(results)
        all_dataset_results[ds_name] = results
        all_dataset_metrics[ds_name] = metrics

        # Print per-dataset summary
        print(f"    {'Method':<18s} {'Acc':>6s} {'O.Gap':>6s} {'InfoU':>6s}")
        for method in METHOD_NAMES:
            m = metrics[method]
            print(f"    {method:<18s} {m['accuracy']:>6.3f} "
                  f"{m['oracle_gap']:>6.3f} {m['info_utilization']:>6.3f}")

    if not loaded_datasets:
        print("  No datasets loaded successfully. Exiting.")
        return {}, {}

    # Statistical tests: Ours vs baselines across all datasets
    print("\n  Cross-dataset statistical tests (Ours vs baselines):")
    for baseline in ["ChatEval", "Unguided Debate", "AgentVerse"]:
        our_accs_all = []
        bl_accs_all = []
        for ds_name in loaded_datasets:
            res = all_dataset_results[ds_name]
            our_a = [1.0 if r["accuracy"] else 0.0
                     for r in res["Ours"] if r["accuracy"] is not None]
            bl_a = [1.0 if r["accuracy"] else 0.0
                    for r in res[baseline] if r["accuracy"] is not None]
            min_len = min(len(our_a), len(bl_a))
            our_accs_all.extend(our_a[:min_len])
            bl_accs_all.extend(bl_a[:min_len])

        if len(our_accs_all) > 1:
            t, p = stats.ttest_rel(our_accs_all, bl_accs_all)
            print(f"    Ours vs {baseline}: t={t:.3f}, p={p:.2e}")

    # ============ IEEE Figure: 2x2 multi-panel ============
    n_ds = len(loaded_datasets)
    ds_short = [DATASET_SHORT.get(d, d[:4]) for d in loaded_datasets]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))

    # (a) Accuracy heatmap: methods × datasets
    ax = axes[0, 0]
    acc_matrix = np.zeros((len(METHOD_NAMES), n_ds))
    for i, method in enumerate(METHOD_NAMES):
        for j, ds_name in enumerate(loaded_datasets):
            acc_matrix[i, j] = all_dataset_metrics[ds_name][method]["accuracy"]
    im = ax.imshow(acc_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_ds))
    ax.set_xticklabels(ds_short, fontsize=6, rotation=30)
    ax.set_yticks(range(len(METHOD_NAMES)))
    ax.set_yticklabels(METHOD_SHORT, fontsize=7)
    for i in range(len(METHOD_NAMES)):
        for j in range(n_ds):
            ax.text(j, i, f"{acc_matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=5.5, fontweight="bold",
                    color="white" if acc_matrix[i,j] < 0.4 else "black")
    ax.set_title("Accuracy: Methods × Datasets")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    add_subfig_label(ax, "a")

    # (b) Per-dataset accuracy comparison (grouped bar)
    ax = axes[0, 1]
    x = np.arange(n_ds)
    bar_width = 0.18
    for i, method in enumerate(METHOD_NAMES):
        accs = [all_dataset_metrics[ds][method]["accuracy"] for ds in loaded_datasets]
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, accs, bar_width, color=METHOD_COLORS[i],
               edgecolor="black", linewidth=0.3, alpha=0.85, label=METHOD_SHORT[i])
    ax.set_xticks(x)
    ax.set_xticklabels(ds_short, fontsize=6, rotation=30)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Dataset")
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "b")

    # (c) Oracle Gap comparison
    ax = axes[1, 0]
    for i, method in enumerate(METHOD_NAMES):
        gaps = [all_dataset_metrics[ds][method]["oracle_gap"] for ds in loaded_datasets]
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, gaps, bar_width, color=METHOD_COLORS[i],
               edgecolor="black", linewidth=0.3, alpha=0.85, label=METHOD_SHORT[i])
    ax.set_xticks(x)
    ax.set_xticklabels(ds_short, fontsize=6, rotation=30)
    ax.set_ylabel("Oracle Gap ↓")
    ax.set_title("Oracle Gap by Dataset (lower is better)")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    add_subfig_label(ax, "c")

    # (d) Difficulty-grouped summary (medium vs hard datasets)
    ax = axes[1, 1]
    medium_ds = [d for d in loaded_datasets if DATASET_REGISTRY[d]["difficulty"] == "medium"]
    hard_ds = [d for d in loaded_datasets if DATASET_REGISTRY[d]["difficulty"] == "hard"]

    categories = ["Medium\nDatasets", "Hard\nDatasets", "All\nDatasets"]
    x_cat = np.arange(len(categories))

    for i, method in enumerate(METHOD_NAMES):
        med_acc = np.mean([all_dataset_metrics[d][method]["accuracy"]
                           for d in medium_ds]) if medium_ds else 0
        hard_acc = np.mean([all_dataset_metrics[d][method]["accuracy"]
                            for d in hard_ds]) if hard_ds else 0
        all_acc = np.mean([all_dataset_metrics[d][method]["accuracy"]
                           for d in loaded_datasets])
        vals = [med_acc, hard_acc, all_acc]
        offset = (i - 1.5) * bar_width
        bars = ax.bar(x_cat + offset, vals, bar_width, color=METHOD_COLORS[i],
                      edgecolor="black", linewidth=0.3, alpha=0.85,
                      label=METHOD_SHORT[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", fontsize=5, fontweight="bold")

    ax.set_xticks(x_cat)
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Performance by Difficulty Group")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "d")

    plt.savefig("figures/exp9_cross_dataset.pdf")
    plt.savefig("figures/exp9_cross_dataset.png", dpi=300)
    plt.close()
    print("\n  → figures/exp9_cross_dataset.{pdf,png}")

    # Print final summary table
    print("\n  " + "=" * 80)
    print(f"  {'Dataset':<15s} " +
          " ".join(f"{m:>10s}" for m in METHOD_SHORT))
    print("  " + "-" * 80)
    for ds_name in loaded_datasets:
        accs = [f"{all_dataset_metrics[ds_name][m]['accuracy']:>10.3f}"
                for m in METHOD_NAMES]
        print(f"  {DATASET_SHORT.get(ds_name, ds_name):<15s} " + " ".join(accs))

    # Average
    avg_accs = []
    for m in METHOD_NAMES:
        avg = np.mean([all_dataset_metrics[d][m]["accuracy"] for d in loaded_datasets])
        avg_accs.append(f"{avg:>10.3f}")
    print("  " + "-" * 80)
    print(f"  {'Average':<15s} " + " ".join(avg_accs))
    print("  " + "=" * 80)

    return all_dataset_results, all_dataset_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default="embedding",
                        choices=["embedding", "bm25", "tfidf"])
    parser.add_argument("--n", type=int, default=50,
                        help="Samples per dataset")
    args = parser.parse_args()
    run(sim_method=args.sim, n_per_dataset=args.n)
