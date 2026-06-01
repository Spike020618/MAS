"""
Experiment 7: Scalability Analysis

论点: Stackelberg 共识机制在智能体规模增长下保持高效收敛
Metrics:
  - Convergence rounds vs N
  - Final consensus energy vs N
  - Communication cost (messages = N²·T)
  - Energy decay rate vs theoretical bound
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import matplotlib.pyplot as plt

from core.stackelberg import StackelbergGame, AgentBid
from experiments.plot_config import (
    ieee_style, DOUBLE_COL, PALETTE, add_subfig_label
)

ieee_style()

AGENT_SCALES = [3, 5, 7, 9, 11, 13, 15]
NUM_ROUNDS = 20


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
    print("Experiment 7: Scalability Analysis")
    print("=" * 60)

    scale_results = {}
    for n in AGENT_SCALES:
        bids = _make_bids(n)
        game = StackelbergGame(num_agents=n, lambda_c=1.0, mu=0.1,
                               eta=0.05, gamma=0.1, R=100.0)
        history = game.run(bids, num_rounds=NUM_ROUNDS)

        conv_round = len(history)
        for step in history:
            if step["energy"] < 1e-4:
                conv_round = step["round"]
                break

        messages = n * (n - 1) * len(history)

        scale_results[n] = {
            "history": history,
            "convergence_rounds": conv_round,
            "final_energy": history[-1]["energy"],
            "initial_energy": history[0]["energy"],
            "messages": messages,
        }
        print(f"  N={n:2d}: conv={conv_round:2d} rounds, "
              f"E={history[-1]['energy']:.2e}, msgs={messages}")

    # Theoretical bound: E(t) <= E(0) * exp(-2γt/(N-1))
    gamma = 0.1

    # ============ IEEE Figure: 2x2 multi-panel ============
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))

    # (a) Energy trajectories for different N
    ax = axes[0, 0]
    for i, n in enumerate(AGENT_SCALES):
        h = scale_results[n]["history"]
        rounds = [s["round"] for s in h]
        energies = [s["energy"] for s in h]
        color = plt.cm.viridis(i / len(AGENT_SCALES))
        ax.semilogy(rounds, energies, "o-", color=color, linewidth=0.9,
                    markersize=2.5, label=f"N={n}")
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Consensus Energy $E$ (log)")
    ax.set_title("Energy Convergence vs Agent Count")
    ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    add_subfig_label(ax, "a")

    # (b) Convergence rounds vs N
    ax = axes[0, 1]
    ns = list(scale_results.keys())
    conv_rounds = [scale_results[n]["convergence_rounds"] for n in ns]
    ax.plot(ns, conv_rounds, "s-", color=PALETTE[0], linewidth=1.2,
            markersize=5, label="Empirical")
    # Theoretical: proportional to (N-1)/(2γ) * ln(E0/ε)
    e0_ref = scale_results[3]["initial_energy"]
    eps = 1e-4
    theory_rounds = [(n - 1) / (2 * gamma) * np.log(max(scale_results[n]["initial_energy"], eps) / eps)
                     for n in ns]
    theory_scale = conv_rounds[0] / max(theory_rounds[0], 1)
    theory_rounds_scaled = [t * theory_scale for t in theory_rounds]
    ax.plot(ns, theory_rounds_scaled, "D--", color=PALETTE[1], linewidth=0.9,
            markersize=4, label="Theoretical bound")
    ax.set_xlabel("Number of Agents $N$")
    ax.set_ylabel("Rounds to Converge")
    ax.set_title("Convergence Speed Scaling")
    ax.legend(fontsize=6)
    add_subfig_label(ax, "b")

    # (c) Communication cost
    ax = axes[1, 0]
    msgs = [scale_results[n]["messages"] for n in ns]
    ax.bar(range(len(ns)), msgs, color=PALETTE[2], edgecolor="black",
           linewidth=0.4, width=0.6)
    # O(N²T) theoretical line
    t_avg = np.mean(conv_rounds)
    theoretical_msgs = [n * (n - 1) * t_avg for n in ns]
    ax.plot(range(len(ns)), theoretical_msgs, "D--", color=PALETTE[1],
            markersize=4, linewidth=0.9, label="$O(N^2 T)$ bound")
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"N={n}" for n in ns], fontsize=6.5)
    ax.set_ylabel("Total Messages")
    ax.set_title("Communication Cost")
    ax.legend(fontsize=6)
    add_subfig_label(ax, "c")

    # (d) Empirical vs theoretical energy decay (N=5 detail)
    ax = axes[1, 1]
    n_detail = 5
    h = scale_results[n_detail]["history"]
    rounds_d = [s["round"] for s in h]
    energies_d = [s["energy"] for s in h]
    e0 = energies_d[0]
    theory_decay = [e0 * np.exp(-2 * gamma * t / (n_detail - 1)) for t in rounds_d]

    ax.semilogy(rounds_d, energies_d, "o-", color=PALETTE[0], linewidth=1.0,
                markersize=4, label="Empirical $E(t)$")
    ax.semilogy(rounds_d, theory_decay, "--", color=PALETTE[1], linewidth=1.0,
                label=r"Bound: $E_0 e^{-2\gamma t/(N{-}1)}$")
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Energy (log)")
    ax.set_title(f"Decay Rate Verification (N={n_detail})")
    ax.legend(fontsize=6)
    ax.fill_between(rounds_d, energies_d, theory_decay, alpha=0.1, color=PALETTE[2])
    add_subfig_label(ax, "d")

    plt.savefig("figures/exp7_scalability.pdf")
    plt.savefig("figures/exp7_scalability.png", dpi=300)
    plt.close()
    print("  → figures/exp7_scalability.{pdf,png}")

    return scale_results


if __name__ == "__main__":
    run()
