"""
Study 1 图表生成 — LNCS 风格（serif/9pt/PDF+PNG），输出至 results/review/figures/

  fig_s1_negativity_bias:   独立评审 accept 率（按分层）vs 人类官方基准
  fig_s1_minority_fate:     讨论后少数派去向（负向棘轮）
  fig_s1_variance_collapse: 评分标准差逐轮坍缩（debate 3 轮箱线图）
  fig_s1_decision_accuracy: 各条件 × 分层 决策一致性

Usage:
  python experiments/study1_figures.py --year 2024
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

PANEL_DIR = "results/review/panel"
STATS_PATH = "results/review/study1_stats.json"
FIG_DIR = "results/review/figures"
STRATA = ["clear_accept", "borderline", "clear_reject"]
STRATA_LABELS = ["Clear\nAccept", "Borderline", "Clear\nReject"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]


def _load(condition: str, year: int) -> list:
    with open(os.path.join(PANEL_DIR, f"{condition}_{year}.json"),
              encoding="utf-8") as f:
        return json.load(f)["papers"]


def _majority_stance(reviews: list):
    votes = [r["stance"] for r in reviews if r["stance"]]
    if not votes:
        return None
    n_acc = votes.count("accept")
    if n_acc == len(votes) - n_acc:
        return "tie"
    return "accept" if n_acc > len(votes) / 2 else "reject"


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


# ── Fig 1: 拒稿偏置 ──────────────────────────────────────────
def fig_negativity_bias(stats_d):
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(STRATA))
    width = 0.36

    llm_rates, llm_err_lo, llm_err_hi = [], [], []
    for s in STRATA:
        d = stats_d["accept_baserate"][s]
        llm_rates.append(100 * d["rate"])
        llm_err_lo.append(100 * (d["rate"] - d["ci_low"]))
        llm_err_hi.append(100 * (d["ci_high"] - d["rate"]))

    # 人类官方基准：clear_accept=100% 被接收，borderline 取该层官方接收率，clear_reject=0%
    parallel = _load("parallel", stats_d["year"])
    human = []
    for s in STRATA:
        dec = [p["decision_binary"] for p in parallel if p["stratum"] == s]
        human.append(100 * np.mean(dec))

    ax.bar(x - width / 2, human, width, color="#BBBBBB", label="Human (official)")
    ax.bar(x + width / 2, llm_rates, width, color=COLORS[2],
           yerr=[llm_err_lo, llm_err_hi], capsize=3,
           label="LLM reviewers (indep.)")
    ax.set_xticks(x)
    ax.set_xticklabels(STRATA_LABELS)
    ax.set_ylabel("Accept rate (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right", frameon=False)
    for xi, r in zip(x, llm_rates):
        ax.text(xi + width / 2, r + 4, f"{r:.1f}%", ha="center", fontsize=7.5)
    _save(fig, "fig_s1_negativity_bias")


# ── Fig 2: 少数派去向 ────────────────────────────────────────
def fig_minority_fate(stats_d):
    flips = stats_d["minority_flips"]
    cats = ["Flipped to\nReject", "Flipped to\nAccept", "Persisted"]
    vals = [flips["to_reject"], flips["to_accept"], flips["persist"]]
    colors = [COLORS[2], COLORS[1], "#999999"]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bars = ax.bar(cats, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, str(v),
                ha="center", fontsize=9)
    ax.set_ylabel("Minority reviewers (count)")
    ax.set_ylim(0, max(vals) * 1.22)
    p = flips["direction_binomial_p"]
    elim = flips["elimination_rate"]
    ax.set_title(f"Directional herding: $p={p:.1e}$\n"
                 f"elimination rate {elim['rate']:.0%} "
                 f"[{elim['ci_low']:.0%}, {elim['ci_high']:.0%}]", fontsize=8.5)
    _save(fig, "fig_s1_minority_fate")


# ── Fig 3: 方差坍缩 ──────────────────────────────────────────
def fig_variance_collapse(stats_d, year):
    debate = _load("debate", year)
    per_round = {0: [], 1: [], 2: []}
    for p in debate:
        for ri, rnd in enumerate(p["rounds"][:3]):
            ratings = [r["rating"] for r in rnd["reviews"] if r["rating"] is not None]
            if len(ratings) >= 2:
                per_round[ri].append(np.std(ratings, ddof=1))

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    data = [per_round[i] for i in range(3)]
    bp = ax.boxplot(data, positions=[1, 2, 3], widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.2),
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor=COLORS[2],
                                   markeredgecolor=COLORS[2], markersize=4))
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS[0])
        patch.set_alpha(0.55)
    means = [np.mean(d) for d in data]
    for i, m in enumerate(means):
        ax.text(i + 1, m + 0.045, f"{m:.2f}", ha="center", fontsize=8,
                color=COLORS[2])
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Round 1\n(indep.)", "Round 2", "Round 3"])
    ax.set_ylabel("Within-panel rating std")
    coll = stats_d["variance_collapse"]["overall"]
    ax.set_title(f"Variance collapse $-{coll['collapse']*100:.0f}\\%$, "
                 f"Wilcoxon $p={coll['wilcoxon_p']:.1e}$", fontsize=8.5)
    _save(fig, "fig_s1_variance_collapse")


# ── Fig 4: 决策一致性 ────────────────────────────────────────
def fig_decision_accuracy(year):
    parallel = _load("parallel", year)
    debate = _load("debate", year)
    star = _load("star", year)

    def acc_by_stratum(papers, round_idx, use_meta=False):
        out = {}
        for s in STRATA:
            vals = []
            for p in papers:
                if p["stratum"] != s:
                    continue
                if use_meta:
                    ms = p.get("meta_review", {}).get("stance")
                else:
                    ms = _majority_stance(p["rounds"][round_idx]["reviews"])
                if ms in ("accept", "reject"):
                    vals.append(int((ms == "accept") == bool(p["decision_binary"])))
            out[s] = 100 * np.mean(vals) if vals else 0
        return out

    conds = [
        ("MajVote\n(indep.)", acc_by_stratum(parallel, 0)),
        ("Panel R1", acc_by_stratum(debate, 0)),
        ("Panel R3", acc_by_stratum(debate, -1)),
        ("Meta-review", acc_by_stratum(star, 0, True)),
    ]

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    x = np.arange(len(STRATA))
    width = 0.19
    for i, (label, accs) in enumerate(conds):
        vals = [accs[s] for s in STRATA]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(STRATA_LABELS)
    ax.set_ylabel("Agreement w/ official (%)")
    ax.set_ylim(0, 118)
    ax.legend(ncol=4, loc="upper center", frameon=False,
              columnspacing=0.8, handlelength=1.2)
    _save(fig, "fig_s1_decision_accuracy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    args = ap.parse_args()

    with open(STATS_PATH, encoding="utf-8") as f:
        stats_d = json.load(f)

    print("Generating Study 1 figures...")
    fig_negativity_bias(stats_d)
    fig_minority_fate(stats_d)
    fig_variance_collapse(stats_d, args.year)
    fig_decision_accuracy(args.year)
    print(f"→ all figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
