"""
Experiment 5: Baseline Comparison — Ours vs ChatEval vs Unguided Debate vs AgentVerse

论点: AEIC + Stackelberg 在准确性、鲁棒性、信息利用率上全面优于现有方法
Baselines:
  1. ChatEval (2023): 均等投票 majority vote
  2. Unguided Debate (Cost of Consensus, 2026): 无引导辩论 + sycophancy
  3. AgentVerse (2023): 中心化 evaluator 裁决

实验设计: 混合 Easy/Hard 任务
  - Easy: 多数Agent有正确答案（高一致性）
  - Hard: 少数Agent有正确答案 / 信息不对称严重（考验共识机制质量）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from experiments.baselines import (
    ChatEvalBaseline, UnguidedDebateBaseline, AgentVerseBaseline, OurMethod
)
from experiments.dataset_loader import load_reasoning_datasets
from experiments.plot_config import (
    ieee_style, DOUBLE_COL, PALETTE, add_subfig_label, significance_marker
)

ieee_style()

METHOD_NAMES = ["ChatEval", "Unguided\nDebate", "AgentVerse", "Ours"]
METHOD_COLORS = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]


def _create_adversarial_tasks(base_dataset, n_adversarial=80, seed=42):
    """
    Create adversarial scenarios where majority is wrong but confident.

    Simulates the failure modes from "The Cost of Consensus":
    - Minority-correct: only 1 of 3 agents has correct reasoning
    - Corrupted majority: 2 agents have plausible but wrong conclusions
    - The wrong agents appear MORE confident (sycophancy trap)
    - The correct agent (agent 2) has less evidence (information asymmetry)

    Three difficulty tiers:
    - Tier 1 (40%): C flipped only — detectable via I→C mismatch
    - Tier 2 (35%): C+I both flipped — requires evidence-level analysis
    - Tier 3 (25%): Fully rationalized — very hard, even for AEIC
    """
    rng = random.Random(seed)
    adversarial = []

    for idx, task in enumerate(base_dataset[:n_adversarial]):
        nodes = [dict(n) for n in task["nodes"]]
        gt = task.get("ground_truth")

        # Determine correct and wrong conclusions
        if gt is True or str(gt).lower() in ("true", "yes", "1"):
            wrong_conclusion = "Based on thorough analysis, the answer is clearly No. The evidence strongly contradicts the proposition with high confidence."
            correct_conclusion = "Based on the evidence, the answer is Yes. Though evidence is limited, the reasoning chain supports an affirmative answer."
            wrong_inference_templates = [
                "Analyzing each fact independently and synthesizing: the collective evidence indicates against the proposition.",
                "Using comparative reasoning, the logical conclusion does not follow from the premises.",
            ]
        else:
            wrong_conclusion = "Based on thorough analysis, the answer is clearly Yes. The evidence strongly supports the proposition with high confidence."
            correct_conclusion = "Based on the evidence, the answer is No. The available facts do not support the claim."
            wrong_inference_templates = [
                "Analyzing each fact independently and synthesizing: the collective evidence strongly suggests the proposition.",
                "Using comparative reasoning, the logical conclusion follows from the premises.",
            ]

        # Assign difficulty tier
        tier_roll = rng.random()
        if tier_roll < 0.40:
            tier = 1  # C flipped only
        elif tier_roll < 0.75:
            tier = 2  # C + I flipped
        else:
            tier = 3  # Fully rationalized (hardest)

        # Agents 0,1: wrong but confident
        nodes[0]["C"] = wrong_conclusion
        nodes[1]["C"] = wrong_conclusion

        if tier >= 2:
            # Also flip I to match wrong C (sycophantic rationalization)
            nodes[0]["I"] = wrong_inference_templates[0]
            nodes[1]["I"] = wrong_inference_templates[1 % len(wrong_inference_templates)]

        if tier >= 3:
            # Fully rationalized: also neutralize correct agent's I
            nodes[2]["I"] = "The evidence is ambiguous and open to multiple interpretations."

        # Agent 2: correct but less confident (minority position)
        nodes[2]["C"] = correct_conclusion

        # Give wrong agents MORE evidence (sycophancy trap for evaluators)
        if isinstance(nodes[0].get("E"), list):
            nodes[0]["E"] = nodes[0]["E"] + ["Additional supporting data point for this analysis"]
        if isinstance(nodes[1].get("E"), list):
            nodes[1]["E"] = nodes[1]["E"] + ["Further corroborating evidence from secondary source"]

        # Reduce correct agent's evidence (information asymmetry)
        if isinstance(nodes[2].get("E"), list) and len(nodes[2]["E"]) > 1:
            nodes[2]["E"] = nodes[2]["E"][:1]

        adversarial.append({**task, "nodes": nodes, "difficulty": "hard", "tier": tier})

    return adversarial


def run(sim_method: str = "embedding"):
    print("\n" + "=" * 60)
    print("Experiment 5: Baseline Comparison")
    print("=" * 60)

    print("  Loading reasoning datasets...")
    dataset = load_reasoning_datasets(n_strategyqa=100, n_hotpotqa=100,
                                       agents_per_task=3, seed=42)

    # Only use tasks with boolean ground truth for accuracy measurement
    # (StrategyQA has boolean; HotpotQA has text answers that can't be yes/no judged)
    boolean_tasks = [t for t in dataset if isinstance(t.get("ground_truth"), bool)]
    print(f"  Boolean-GT tasks (StrategyQA): {len(boolean_tasks)}")

    # Split: half normal (easy), half adversarial (hard)
    mid = len(boolean_tasks) // 2
    easy_tasks = boolean_tasks[:mid]
    hard_tasks = _create_adversarial_tasks(boolean_tasks[mid:], n_adversarial=len(boolean_tasks) - mid, seed=42)
    all_tasks = easy_tasks + hard_tasks
    print(f"  Tasks: {len(easy_tasks)} easy + {len(hard_tasks)} hard = {len(all_tasks)} total")

    # Initialize methods
    chateval = ChatEvalBaseline(sim_method=sim_method)
    debate = UnguidedDebateBaseline(sim_method=sim_method, seed=42)
    agentverse = AgentVerseBaseline(sim_method=sim_method)
    ours = OurMethod(sim_method=sim_method)

    # Collect results
    results = {
        "ChatEval": [], "Unguided Debate": [],
        "AgentVerse": [], "Ours": [],
    }
    results_easy = {m: [] for m in results}
    results_hard = {m: [] for m in results}

    print("  Running 4 methods on all tasks...")
    for i, task in enumerate(all_tasks):
        gt = task.get("ground_truth")
        nodes = task["nodes"]
        difficulty = task.get("difficulty", "easy")

        r_ce = chateval.consensus(nodes, gt)
        r_ud = debate.consensus(nodes, gt)
        r_av = agentverse.consensus(nodes, gt)
        r_ours = ours.consensus(nodes, gt)

        results["ChatEval"].append(r_ce)
        results["Unguided Debate"].append(r_ud)
        results["AgentVerse"].append(r_av)
        results["Ours"].append(r_ours)

        target = results_easy if difficulty == "easy" else results_hard
        target["ChatEval"].append(r_ce)
        target["Unguided Debate"].append(r_ud)
        target["AgentVerse"].append(r_av)
        target["Ours"].append(r_ours)

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(all_tasks)} tasks")

    # Compute aggregate metrics
    def compute_metrics(res_dict):
        metrics = {}
        for method, res_list in res_dict.items():
            accs = [1.0 if r["accuracy"] else 0.0 for r in res_list if r["accuracy"] is not None]
            sims = [r["similarity"] for r in res_list]
            strengths = [r["consensus_strength"] for r in res_list]
            oracle_gaps = [r["oracle_gap"] for r in res_list]
            info_utils = [r["info_utilization"] for r in res_list]
            metrics[method] = {
                "accuracy": np.mean(accs) if accs else 0.0,
                "accuracy_ci": 1.96 * np.std(accs) / np.sqrt(max(len(accs), 1)) if accs else 0.0,
                "similarity": np.mean(sims) if sims else 0.0,
                "consensus_strength": np.mean(strengths) if strengths else 0.0,
                "oracle_gap": np.mean(oracle_gaps) if oracle_gaps else 0.0,
                "info_utilization": np.mean(info_utils) if info_utils else 0.0,
            }
        return metrics

    metrics_all = compute_metrics(results)
    metrics_easy = compute_metrics(results_easy)
    metrics_hard = compute_metrics(results_hard)

    # Print results
    print(f"\n  {'Method':<20s} {'All Acc':>8s} {'Easy Acc':>9s} {'Hard Acc':>9s} "
          f"{'Oracle Gap':>11s} {'Info Util':>10s}")
    print("  " + "-" * 75)
    for method in ["ChatEval", "Unguided Debate", "AgentVerse", "Ours"]:
        print(f"  {method:<20s} {metrics_all[method]['accuracy']:>8.3f} "
              f"{metrics_easy[method]['accuracy']:>9.3f} "
              f"{metrics_hard[method]['accuracy']:>9.3f} "
              f"{metrics_all[method]['oracle_gap']:>11.3f} "
              f"{metrics_all[method]['info_utilization']:>10.3f}")

    # Statistical tests
    methods_order = ["ChatEval", "Unguided Debate", "AgentVerse", "Ours"]
    our_accs = [1.0 if r["accuracy"] else 0.0 for r in results["Ours"] if r["accuracy"] is not None]
    p_values = {}
    for baseline in ["ChatEval", "Unguided Debate", "AgentVerse"]:
        bl_accs = [1.0 if r["accuracy"] else 0.0 for r in results[baseline] if r["accuracy"] is not None]
        min_len = min(len(our_accs), len(bl_accs))
        if min_len > 1:
            t, p = stats.ttest_rel(our_accs[:min_len], bl_accs[:min_len])
            if not np.isnan(p):
                p_values[baseline] = p
                print(f"  t-test Ours vs {baseline}: t={t:.3f}, p={p:.2e}")

    # ============ IEEE Figure: 2x2 multi-panel ============
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))

    # (a) Accuracy: Easy vs Hard breakdown
    ax = axes[0, 0]
    x = np.arange(4)
    width = 0.35
    easy_accs = [metrics_easy[m]["accuracy"] for m in methods_order]
    hard_accs = [metrics_hard[m]["accuracy"] for m in methods_order]
    bars1 = ax.bar(x - width / 2, easy_accs, width, label="Easy (majority correct)",
                   color=[c for c in METHOD_COLORS], edgecolor="black", linewidth=0.4, alpha=0.6)
    bars2 = ax.bar(x + width / 2, hard_accs, width, label="Hard (minority correct)",
                   color=[c for c in METHOD_COLORS], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_NAMES, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Task Difficulty")
    ax.legend(fontsize=6, loc="lower left")
    ax.set_ylim(0, 1.15)
    for bar, acc in zip(bars2, hard_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2f}", ha="center", fontsize=6, fontweight="bold")
    add_subfig_label(ax, "a")

    # (b) Robustness metrics: Oracle Gap + Vulnerability
    ax = axes[0, 1]
    hard_oracle_gaps = [metrics_hard[m]["oracle_gap"] for m in methods_order]
    all_info_utils = [metrics_all[m]["info_utilization"] for m in methods_order]
    bars1 = ax.bar(x - width / 2, hard_oracle_gaps, width, label="Oracle Gap (Hard) ↓",
                   color=METHOD_COLORS, edgecolor="black", linewidth=0.4, alpha=0.6)
    bars2 = ax.bar(x + width / 2, all_info_utils, width, label="Info Utilization ↑",
                   color=METHOD_COLORS, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_NAMES, fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("Robustness: Oracle Gap vs Info Utilization")
    ax.legend(fontsize=6, loc="upper left")
    for bar, v in zip(bars1, hard_oracle_gaps):
        if v > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", fontsize=5.5)
    add_subfig_label(ax, "b")

    # (c) Consensus Similarity comparison
    ax = axes[1, 0]
    easy_sims = [metrics_easy[m]["similarity"] for m in methods_order]
    hard_sims = [metrics_hard[m]["similarity"] for m in methods_order]
    bars1 = ax.bar(x - width / 2, easy_sims, width, label="Easy",
                   color=METHOD_COLORS, edgecolor="black", linewidth=0.4, alpha=0.6)
    bars2 = ax.bar(x + width / 2, hard_sims, width, label="Hard",
                   color=METHOD_COLORS, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_NAMES, fontsize=7)
    ax.set_ylabel("Consensus Similarity")
    ax.set_title("Semantic Coherence by Difficulty")
    ax.legend(fontsize=6)
    # Annotate gap (coherence drop under adversarial)
    for i in range(4):
        drop = easy_sims[i] - hard_sims[i]
        if drop > 0.01:
            mid_y = (easy_sims[i] + hard_sims[i]) / 2
            ax.annotate(f"Δ={drop:.2f}", xy=(i + 0.2, mid_y),
                        fontsize=5.5, color="red", ha="center")
    add_subfig_label(ax, "c")

    # (d) Multi-metric overall comparison
    ax = axes[1, 1]
    metric_labels = ["Accuracy", "Sim. Quality", "Consensus\nStrength", "Info Util."]
    metric_keys = ["accuracy", "similarity", "consensus_strength", "info_utilization"]
    n_metrics = len(metric_labels)
    x_m = np.arange(n_metrics)
    bar_width = 0.18
    for i, method in enumerate(methods_order):
        vals = [metrics_all[method][k] for k in metric_keys]
        offset = (i - 1.5) * bar_width
        ax.bar(x_m + offset, vals, bar_width, color=METHOD_COLORS[i],
               edgecolor="black", linewidth=0.3, alpha=0.85,
               label=method.replace("\n", " "))
    ax.set_xticks(x_m)
    ax.set_xticklabels(metric_labels, fontsize=6.5)
    ax.set_ylabel("Score")
    ax.set_title("Overall Multi-Dimensional Comparison")
    ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.15)
    add_subfig_label(ax, "d")

    plt.tight_layout()
    plt.savefig("figures/exp5_baseline_comparison.pdf")
    plt.savefig("figures/exp5_baseline_comparison.png", dpi=300)
    plt.close()
    print("\n  → figures/exp5_baseline_comparison.{pdf,png}")

    return results, metrics_all, metrics_easy, metrics_hard


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default="embedding", choices=["embedding", "bm25", "tfidf"])
    args = parser.parse_args()
    run(sim_method=args.sim)
