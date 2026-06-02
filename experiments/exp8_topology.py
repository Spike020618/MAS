"""
Experiment 8: Network Topology Comparison

论点: Stackelberg 共识机制在不同网络拓扑下均可收敛，
     收敛速度与拉普拉斯代数连通度 λ₂ 正相关。

Topologies:
  - Complete: 完全图 (上界，最快收敛)
  - Ring: 环形 (下界，最慢收敛)
  - Star: 星形 (中心化极端)
  - Random: Erdős–Rényi G(N, 0.5)
  - Small-World: Watts-Strogatz (高聚类 + 短路径)

Metrics:
  - 收敛轮数 (energy < threshold)
  - 最终共识能量
  - 通信消息量 (基于实际边数)
  - λ₂ 与收敛速度的相关性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from core.stackelberg import StackelbergGame, AgentBid
from core.topology import (
    make_topology, laplacian_lambda2, topology_summary,
    TOPOLOGY_NAMES, TOPOLOGY_LABELS,
)
from experiments.plot_config import (
    ieee_style, DOUBLE_COL, PALETTE, add_subfig_label
)

ieee_style()

NUM_AGENTS = 7
NUM_ROUNDS = 30
CONVERGENCE_THRESHOLD = 1e-4


def _make_bids(n: int, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    return [
        AgentBid(
            agent_id=f"agent-{i}",
            model=f"model-{i}",
            alignment_willingness=float(rng.uniform(0.4, 0.9)),
            revision_cost=float(rng.uniform(8.0, 25.0)),
            max_revision_depth=float(rng.uniform(0.5, 0.95)),
            historical_alignment=float(rng.uniform(0.6, 0.9)),
        )
        for i in range(n)
    ]


def run():
    print("\n" + "=" * 60)
    print("Experiment 8: Network Topology Comparison")
    print("=" * 60)

    bids = _make_bids(NUM_AGENTS)
    results = {}

    for topo_name in TOPOLOGY_NAMES:
        W = make_topology(topo_name, NUM_AGENTS)
        summary = topology_summary(W)
        lam2 = summary["lambda2"]

        game = StackelbergGame(
            num_agents=NUM_AGENTS, lambda_c=1.0, mu=0.1,
            eta=0.05, gamma=0.1, R=100.0, W=W,
        )
        history = game.run(bids, num_rounds=NUM_ROUNDS)

        # Find convergence round
        conv_round = NUM_ROUNDS
        for step in history:
            if step["energy"] < CONVERGENCE_THRESHOLD:
                conv_round = step["round"]
                break

        # Communication cost: messages per round = number of edges * 2
        n_edges = summary["edges"]
        total_msgs = n_edges * 2 * len(history)

        results[topo_name] = {
            "history": history,
            "convergence_rounds": conv_round,
            "final_energy": history[-1]["energy"],
            "initial_energy": history[0]["energy"] if history else 0,
            "lambda2": lam2,
            "summary": summary,
            "total_messages": total_msgs,
            "final_theta": history[-1]["theta"] if history else np.zeros(NUM_AGENTS),
        }

        label = TOPOLOGY_LABELS[topo_name]
        print(f"  {label:15s}: conv={conv_round:2d} rounds, "
              f"E_final={history[-1]['energy']:.2e}, "
              f"λ₂={lam2:.4f}, msgs={total_msgs}")

    # λ₂ vs convergence speed correlation
    lam2_vals = [results[t]["lambda2"] for t in TOPOLOGY_NAMES]
    conv_vals = [results[t]["convergence_rounds"] for t in TOPOLOGY_NAMES]
    if len(set(conv_vals)) > 1:
        r_corr, p_corr = stats.pearsonr(lam2_vals, conv_vals)
        print(f"\n  λ₂ vs convergence rounds: r={r_corr:.3f}, p={p_corr:.3f}")
    else:
        r_corr, p_corr = -1.0, 0.0
        print(f"\n  λ₂ vs convergence rounds: all converged at same round")

    # ============ IEEE Figure: 2x2 multi-panel ============
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))
    topo_colors = [PALETTE[i % len(PALETTE)] for i in range(len(TOPOLOGY_NAMES))]
    topo_labels = [TOPOLOGY_LABELS[t] for t in TOPOLOGY_NAMES]

    # (a) Energy decay trajectories for all topologies
    ax = axes[0, 0]
    for i, topo_name in enumerate(TOPOLOGY_NAMES):
        h = results[topo_name]["history"]
        rounds = [s["round"] for s in h]
        energies = [s["energy"] for s in h]
        ax.semilogy(rounds, energies, "o-", color=topo_colors[i],
                    linewidth=1.0, markersize=2.5, label=topo_labels[i])
    ax.axhline(CONVERGENCE_THRESHOLD, color="gray", linestyle="--",
               linewidth=0.6, label=f"ε={CONVERGENCE_THRESHOLD}")
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Consensus Energy $E$ (log)")
    ax.set_title("Energy Convergence by Topology")
    ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    add_subfig_label(ax, "a")

    # (b) Convergence rounds & λ₂ comparison
    ax = axes[0, 1]
    x = np.arange(len(TOPOLOGY_NAMES))
    width = 0.35
    bars1 = ax.bar(x - width/2, conv_vals, width, label="Conv. Rounds",
                   color=topo_colors, edgecolor="black", linewidth=0.4)
    ax2 = ax.twinx()
    ax2.plot(x, lam2_vals, "D-", color="black", linewidth=1.2,
             markersize=5, label="$\\lambda_2$")
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, fontsize=6, rotation=15)
    ax.set_ylabel("Convergence Rounds")
    ax2.set_ylabel("Algebraic Connectivity $\\lambda_2$")
    ax.set_title("Convergence Speed vs Connectivity")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper left")
    for bar, v in zip(bars1, conv_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha="center", fontsize=6, fontweight="bold")
    add_subfig_label(ax, "b")

    # (c) Communication cost comparison
    ax = axes[1, 0]
    msgs = [results[t]["total_messages"] for t in TOPOLOGY_NAMES]
    edges = [results[t]["summary"]["edges"] for t in TOPOLOGY_NAMES]
    bars = ax.bar(x, msgs, width=0.5, color=topo_colors,
                  edgecolor="black", linewidth=0.4)
    for bar, m, e in zip(bars, msgs, edges):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{m}\n({e}E)", ha="center", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels, fontsize=6, rotation=15)
    ax.set_ylabel("Total Messages")
    ax.set_title("Communication Cost by Topology")
    add_subfig_label(ax, "c")

    # (d) Final θ distribution per topology
    ax = axes[1, 1]
    bp_data = [results[t]["final_theta"] for t in TOPOLOGY_NAMES]
    bp = ax.boxplot(bp_data, tick_labels=topo_labels, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markersize=4, markerfacecolor="black"))
    for patch, color in zip(bp["boxes"], topo_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Final $\\theta_i$")
    ax.set_title("Parameter Consensus Quality")
    ax.tick_params(axis="x", labelsize=6, rotation=15)
    # Add θ spread annotation
    for i, topo_name in enumerate(TOPOLOGY_NAMES):
        theta = results[topo_name]["final_theta"]
        spread = np.std(theta)
        ax.text(i + 1, np.max(theta) + 0.005, f"σ={spread:.4f}",
                ha="center", fontsize=5, color="red")
    add_subfig_label(ax, "d")

    plt.savefig("figures/exp8_topology.pdf")
    plt.savefig("figures/exp8_topology.png", dpi=300)
    plt.close()
    print("\n  → figures/exp8_topology.{pdf,png}")

    return results


if __name__ == "__main__":
    run()
