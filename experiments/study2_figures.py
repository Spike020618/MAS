"""
论文图：跨模型对比 + Study 2 干预总结（LNCS 风格，serif 9pt，PDF+PNG）

输出：
  experiments/results/review/figures/fig_s2_crossmodel.{pdf,png}
  experiments/results/review/figures/fig_s3_interventions.{pdf,png}
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150,
})
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
FIG_DIR = "experiments/results/review/figures"


def fig_crossmodel():
    with open("experiments/results/review/cross_model_analysis.json", encoding="utf-8") as f:
        d = json.load(f)
    models = [("deepseek", "DeepSeek-V3"), ("glm", "GLM-4-Flash"), ("r1", "DeepSeek-R1")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))

    # (a) accept 基准率（r1，按分层）
    ax = axes[0]
    strata = ["clear_accept", "borderline", "clear_reject"]
    x = np.arange(3)
    w = 0.26
    for k, (key, name) in enumerate(models):
        vals = [d[key]["accept_baserate_r1"][s] * 100 for s in strata]
        ax.bar(x + (k - 1) * w, vals, w, color=COLORS[k], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["clear\naccept", "borderline", "clear\nreject"])
    ax.set_ylabel("Accept rate R1 (%)")
    ax.set_title("(a) Independent accept rate")
    ax.legend(frameon=False, loc="lower left")
    ax.set_ylim(0, 109)

    # (b) 消灭率 + 翻转方向
    ax = axes[1]
    elim = [d[k]["minority"]["elimination_rate"] * 100 for k, _ in models]
    torej = [(d[k]["minority"]["flip_to_reject_share"] or 0) * 100
             for k, _ in models]
    x = np.arange(2)
    for k, (_, name) in enumerate(models):
        ax.bar(x + (k - 1) * w, [elim[k], torej[k]], w, color=COLORS[k],
               label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["minority\nelimination", "flips toward\nreject"])
    ax.set_ylabel("(%)")
    ax.set_title("(b) Minority fate")
    ax.set_ylim(0, 119)

    # (c) 方差坍缩 vs 决策一致率
    ax = axes[2]
    collapse = [d[k]["variance_collapse"] * 100 for k, _ in models]
    acc = [d[k]["decision_acc_r3"] * 100 for k, _ in models]
    x = np.arange(2)
    for k, (_, name) in enumerate(models):
        ax.bar(x + (k - 1) * w, [collapse[k], acc[k]], w, color=COLORS[k])
    ax.set_xticks(x)
    ax.set_xticklabels(["variance\ncollapse", "decision\nagreement R3"])
    ax.set_ylabel("(%)")
    ax.set_title("(c) Collapse vs. decision quality")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_s2_crossmodel.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  fig_s2_crossmodel done")


def fig_interventions():
    s2a = json.load(open("experiments/results/review/study2a_theta_star.json",
                         encoding="utf-8"))
    s2b = json.load(open("experiments/results/review/regulated_v3_analysis.json",
                         encoding="utf-8"))
    s2c = json.load(open("experiments/results/review/study2c_routing.json",
                         encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    # (a) 2a：校准聚合 clear_accept 修复
    ax = axes[0]
    t = s2a["test"]
    methods = [("majority\nvote", t["majority_vote"]),
               ("mean\nthreshold", t["mean_threshold"]),
               ("$\\theta^*$\naggregation", t["theta_star"])]
    x = np.arange(3)
    overall = [m[1]["agreement"] * 100 for m in methods]
    ca = [m[1]["agree[clear_accept]"] * 100 for m in methods]
    w = 0.36
    ax.bar(x - w / 2, overall, w, color=COLORS[0], label="overall")
    ax.bar(x + w / 2, ca, w, color=COLORS[2], label="clear-accept stratum")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods], fontsize=7.5)
    ax.set_ylabel("Agreement (%)")
    ax.set_title("(a) 2a: calibrated aggregation")
    ax.legend(frameon=False)

    # (b) 2b：行为调控（消灭率 + 决策安全）
    ax = axes[1]
    fp = s2b["full_panel"]
    elim = [fp["unregulated"]["minority"]["elimination_rate"] * 100,
            fp["regulated"]["minority"]["elimination_rate"] * 100]
    acc = [fp["unregulated"]["decision_acc_r3"] * 100,
           fp["regulated"]["decision_acc_r3"] * 100]
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, elim, w, color=COLORS[3], label="minority elimination")
    ax.bar(x + w / 2, acc, w, color=COLORS[1], label="decision agreement")
    ax.set_xticks(x)
    ax.set_xticklabels(["unregulated", "regulated"])
    ax.set_ylabel("(%)")
    ax.set_title("(b) 2b: behavioral regulation")
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0, 100)

    # (c) 2c：成本-准确率
    ax = axes[2]
    t = s2c["test"]
    pts = [("fixed indep. ($\\theta^*$)", t["fixed_theta_indep"], COLORS[0], "s"),
           ("fixed debate", t["fixed_debate"], COLORS[2], "o"),
           ("fixed meta", t["fixed_meta"], COLORS[3], "^"),
           ("router", t["router_argmax"], COLORS[1], "*"),
           ("oracle", t["oracle"], "gray", "P")]
    for name, d, c, mk in pts:
        ax.scatter(d["avg_cost"], d["agreement"] * 100, c=c, marker=mk,
                   s=90 if mk == "*" else 45, label=name,
                   edgecolor="k" if mk == "*" else "none", linewidth=0.5,
                   zorder=3)
    ax.set_xlabel("Avg. LLM calls / paper")
    ax.set_ylabel("Agreement (%)")
    ax.set_title("(c) 2c: cost–accuracy routing")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_s3_interventions.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  fig_s3_interventions done")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_crossmodel()
    fig_interventions()
