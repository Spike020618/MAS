"""
Experiment 3: Stackelberg Game Convergence Analysis

论点: Stackelberg激励机制驱动共识能量单调递减并保证收敛
Metrics:
  - Consensus energy E over rounds
  - Leader utility J over rounds
  - θ parameter trajectories
  - Convergence speed under different λ (consensus pressure)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core.stackelberg import StackelbergGame, AgentBid
from experiments.plot_config import (
    ieee_style, DOUBLE_COL, PALETTE, add_subfig_label
)

ieee_style()

AGENT_CONFIGS = [
    ("Agent-A (GPT-4)", "gpt-4", 0.8, 15.0, 0.9, 0.85),
    ("Agent-B (Claude)", "claude-3", 0.6, 20.0, 0.7, 0.75),
    ("Agent-C (Gemini)", "gemini", 0.7, 12.0, 0.8, 0.80),
    ("Agent-D (Llama)", "llama-3", 0.5, 25.0, 0.6, 0.70),
    ("Agent-E (Mistral)", "mistral", 0.75, 10.0, 0.85, 0.82),
]


def create_bids(n=5):
    return [
        AgentBid(agent_id=c[0], model=c[1], alignment_willingness=c[2],
                 revision_cost=c[3], max_revision_depth=c[4],
                 historical_alignment=c[5])
        for c in AGENT_CONFIGS[:n]
    ]


def run():
    print("\n" + "=" * 60)
    print("Experiment 3: Stackelberg Game Convergence")
    print("=" * 60)

    bids = create_bids(5)
    num_rounds = 15

    # Main experiment
    game = StackelbergGame(num_agents=5, lambda_c=1.0, mu=0.1, eta=0.05, gamma=0.1, R=100.0)
    history = game.run(bids, num_rounds=num_rounds)

    print(f"  Rounds: {len(history)}")
    print(f"  Energy: {history[0]['energy']:.6f} → {history[-1]['energy']:.6f}")
    print(f"  Utility: {history[0]['utility']:.2f} → {history[-1]['utility']:.2f}")
    print(f"  Final θ: {history[-1]['theta']}")

    # λ sweep experiment
    lambdas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    lambda_histories = {}
    for lam in lambdas:
        g = StackelbergGame(num_agents=5, lambda_c=lam, mu=0.1, eta=0.05, gamma=0.1, R=100.0)
        lambda_histories[lam] = g.run(bids, num_rounds=num_rounds)

    # η (learning rate) sweep
    etas = [0.01, 0.02, 0.05, 0.1, 0.2]
    eta_histories = {}
    for eta in etas:
        g = StackelbergGame(num_agents=5, lambda_c=1.0, mu=0.1, eta=eta, gamma=0.1, R=100.0)
        eta_histories[eta] = g.run(bids, num_rounds=num_rounds)

    # ============ IEEE Figure: 2x2 multi-panel ============
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.72))

    # (a) Energy & Utility dual-axis
    ax1 = axes[0, 0]
    rounds = [h["round"] for h in history]
    energies = [h["energy"] for h in history]
    utilities = [h["utility"] for h in history]

    ln1 = ax1.plot(rounds, energies, "o-", color=PALETTE[1], linewidth=1.0,
                   markersize=3, label="Energy $E$")
    ax1.set_xlabel("Round $t$")
    ax1.set_ylabel("Consensus Energy $E$", color=PALETTE[1])
    ax1.tick_params(axis="y", labelcolor=PALETTE[1])

    ax2 = ax1.twinx()
    ln2 = ax2.plot(rounds, utilities, "s--", color=PALETTE[0], linewidth=1.0,
                   markersize=3, label="Utility $J$")
    ax2.set_ylabel("Leader Utility $J$", color=PALETTE[0])
    ax2.tick_params(axis="y", labelcolor=PALETTE[0])

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=6, loc="center right")
    ax1.set_title("Convergence: Energy ↓, Utility ↑")
    add_subfig_label(ax1, "a")

    # (b) θ trajectories for each agent
    ax = axes[0, 1]
    theta_hist = np.array([h["theta"] for h in history])
    for i in range(min(5, theta_hist.shape[1])):
        ax.plot(rounds, theta_hist[:, i], "-", color=PALETTE[i % 5],
                linewidth=1.0, label=AGENT_CONFIGS[i][0].split(" ")[0],
                marker=["o", "s", "^", "D", "v"][i], markersize=3, markevery=2)
    ax.axhline(np.mean(theta_hist[-1]), color="gray", linestyle="--",
               linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Incentive Parameter $\\theta_i$")
    ax.set_title("Agent Parameter Trajectories")
    ax.legend(fontsize=6, ncol=2, loc="best")
    add_subfig_label(ax, "b")

    # (c) λ sweep - convergence speed (log scale)
    ax = axes[1, 0]
    for i, lam in enumerate(lambdas):
        h = lambda_histories[lam]
        r = [x["round"] for x in h]
        e = [x["energy"] for x in h]
        if max(e) > 0:
            e_normalized = [x / max(e[0], 1e-10) for x in e]
            ax.semilogy(r, e_normalized, "-", color=plt.cm.viridis(i / len(lambdas)),
                        linewidth=0.9, label=f"$\\lambda$={lam}")
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Normalized Energy $E/E_0$ (log)")
    ax.set_title("Convergence Speed vs. $\\lambda$")
    ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    add_subfig_label(ax, "c")

    # (d) η sweep - final energy and convergence
    ax = axes[1, 1]
    final_energies = [eta_histories[eta][-1]["energy"] for eta in etas]
    convergence_rounds = []
    for eta in etas:
        h = eta_histories[eta]
        threshold = h[0]["energy"] * 0.01
        conv_r = len(h)
        for step in h:
            if step["energy"] < threshold:
                conv_r = step["round"]
                break
        convergence_rounds.append(conv_r)

    ax_bar = ax
    x = np.arange(len(etas))
    bars = ax_bar.bar(x, final_energies, width=0.4, color=PALETTE[0],
                      edgecolor="black", linewidth=0.5, alpha=0.7, label="Final Energy")
    ax_bar.set_ylabel("Final Energy", color=PALETTE[0])
    ax_bar.tick_params(axis="y", labelcolor=PALETTE[0])

    ax_line = ax_bar.twinx()
    ax_line.plot(x, convergence_rounds, "D-", color=PALETTE[1], markersize=4,
                 linewidth=1.0, label="Conv. Rounds")
    ax_line.set_ylabel("Rounds to Converge", color=PALETTE[1])
    ax_line.tick_params(axis="y", labelcolor=PALETTE[1])

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"η={e}" for e in etas], fontsize=6.5)
    ax_bar.set_title("Learning Rate $\\eta$ Sensitivity")

    lns_b = [bars] + ax_line.get_lines()
    labs_b = ["Final Energy", "Conv. Rounds"]
    ax_bar.legend(lns_b, labs_b, fontsize=6, loc="upper left")
    add_subfig_label(ax, "d")

    plt.savefig("figures/exp3_stackelberg.pdf")
    plt.savefig("figures/exp3_stackelberg.png", dpi=300)
    plt.close()
    print("  → figures/exp3_stackelberg.{pdf,png}")

    return history, lambda_histories, eta_histories


if __name__ == "__main__":
    run()
