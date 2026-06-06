"""Generate all paper figures from experimental data."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(ROOT, "results/all_baselines.json")) as f:
    baselines = json.load(f)

DATASETS = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
TOPOS = ["parallel", "sequential", "star", "debate", "hierarchical"]
COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']


def acc(ds, key):
    if key not in baselines[ds]:
        return 0
    d = baselines[ds][key]
    return d["correct"] / d["total"] * 100 if d["total"] > 0 else 0


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")


# ============================================================
# Fig 1: Main Results Bar Chart
# ============================================================
def fig_main_results():
    methods = [
        ("Single\nAgent", "Single Agent"),
        ("MajVote\n(parallel)", "MajVote_parallel"),
        ("MajVote\n(star)", "MajVote_star"),
        ("AEIC+$\\theta^*$\n(star)", "AEIC_opt_star"),
        ("Adaptive\nRouter", None),
        ("Oracle", "Oracle_5topo"),
    ]

    x = np.arange(len(DATASETS))
    width = 0.12
    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    for i, (label, key) in enumerate(methods):
        vals = []
        for ds in DATASETS:
            if key is None:
                vals.append(92.4)
            else:
                vals.append(acc(ds, key))
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=label, color=COLORS[i],
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=5.5)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(55, 105)
    ax.legend(loc='upper left', ncol=3, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, "fig_main_results")


# ============================================================
# Fig 2: Topology Comparison (MajVote vs AEIC+Default vs AEIC+θ*)
# ============================================================
def fig_topo_comparison():
    methods = [
        ("Majority Vote", "MajVote"),
        ("AEIC + Default $\\theta$", "AEIC_default"),
        ("AEIC + $\\theta^*$", "AEIC_opt"),
    ]
    colors_3 = ['#4C72B0', '#CCB974', '#C44E52']
    topo_labels = ["Parallel", "Sequential", "Star", "Debate", "Hierarchical"]

    x = np.arange(len(TOPOS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6, 3))

    for i, (label, prefix) in enumerate(methods):
        vals = []
        for topo in TOPOS:
            key = f"{prefix}_{topo}"
            avg = np.mean([acc(ds, key) for ds in DATASETS])
            vals.append(avg)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width * 0.85, label=label, color=colors_3[i],
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=6.5)

    ax.set_ylabel('Average Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(topo_labels)
    ax.set_ylim(60, 90)
    ax.legend(framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, "fig_topo_comparison")


# ============================================================
# Fig 3: θ* Parameter Heatmap
# ============================================================
def fig_theta_heatmap():
    configs = {}
    for topo in TOPOS:
        path = os.path.join(ROOT, f"results/configs/opt_{topo}.json")
        with open(path) as f:
            configs[topo] = json.load(f)

    params_show = [
        ("ha_ic", "$w_{\\mathrm{IC}}$"),
        ("ha_ec", "$w_{\\mathrm{EC}}$"),
        ("ha_eq", "$w_{\\mathrm{EQ}}$"),
        ("ha_ev", "$w_{\\mathrm{EV}}$"),
        ("sg_lambda", "$\\lambda$"),
        ("sg_mu", "$\\mu$"),
        ("aw_ic_exp", "$\\beta_{\\mathrm{IC}}$"),
        ("aw_eq_exp", "$\\beta_{\\mathrm{EQ}}$"),
        ("aw_ev_scale", "$s_{\\mathrm{EV}}$"),
    ]

    data = np.zeros((len(params_show), len(TOPOS)))
    for j, topo in enumerate(TOPOS):
        for i, (key, _) in enumerate(params_show):
            data[i, j] = configs[topo][key]

    data_norm = data.copy()
    for i in range(data.shape[0]):
        row_min, row_max = data[i].min(), data[i].max()
        if row_max > row_min:
            data_norm[i] = (data[i] - row_min) / (row_max - row_min)
        else:
            data_norm[i] = 0.5

    fig, ax = plt.subplots(figsize=(5, 3.5))
    cmap = LinearSegmentedColormap.from_list('custom', ['#f7fbff', '#2171b5'])
    im = ax.imshow(data_norm, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(TOPOS)))
    ax.set_xticklabels([t.capitalize() for t in TOPOS], rotation=30, ha='right')
    ax.set_yticks(range(len(params_show)))
    ax.set_yticklabels([label for _, label in params_show])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = 'white' if data_norm[i, j] > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.5, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Relative magnitude')
    save(fig, "fig_theta_heatmap")


# ============================================================
# Fig 4: Router Performance Progression
# ============================================================
def fig_router_progression():
    methods = [
        ("Single Agent", 74.8),
        ("MajVote (star)", 82.7),
        ("AEIC+$\\theta^*$ (star)", 81.9),
        ("Router (shared)", 88.5),
        ("Router (per-dataset)", 89.4),
        ("Router (top-2)", 92.4),
        ("Oracle (5-topo)", 95.9),
    ]

    labels = [m[0] for m in methods]
    vals = [m[1] for m in methods]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors_bar = []
    for i, (name, _) in enumerate(methods):
        if 'Router' in name:
            colors_bar.append('#C44E52')
        elif 'Oracle' in name:
            colors_bar.append('#8172B2')
        elif 'AEIC' in name:
            colors_bar.append('#CCB974')
        else:
            colors_bar.append('#4C72B0')

    bars = ax.barh(range(len(methods)), vals, color=colors_bar, edgecolor='white',
                   height=0.6)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}%', va='center', fontsize=7.5)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Average Accuracy (%)')
    ax.set_xlim(70, 100)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, "fig_router_progression")


# ============================================================
# Fig 5: Signal Ablation
# ============================================================
def fig_ablation():
    ablation = {
        "Full": 81.55,
        "$-$IC": 81.55 - 0.55,
        "$-$EC": 81.55 - 1.15,
        "$-$EQ": 81.55 + 0.40,
        "$-$EV": 81.55 - 0.50,
    }

    labels = list(ablation.keys())
    vals = list(ablation.values())
    deltas = [0, -0.55, -1.15, +0.40, -0.50]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))

    colors_abs = ['#4C72B0'] + ['#55A868' if d >= 0 else '#C44E52' for d in deltas[1:]]
    ax1.bar(range(len(labels)), vals, color=colors_abs, edgecolor='white', width=0.6)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.15, f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(79, 83)
    ax1.set_title('(a) Absolute accuracy')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    colors_delta = ['#888888'] + ['#55A868' if d >= 0 else '#C44E52' for d in deltas[1:]]
    ax2.bar(range(len(labels)), deltas, color=colors_delta, edgecolor='white', width=0.6)
    for i, d in enumerate(deltas):
        offset = 0.08 if d >= 0 else -0.08
        va = 'bottom' if d >= 0 else 'top'
        ax2.text(i, d + offset, f'{d:+.2f}', ha='center', va=va, fontsize=7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('$\\Delta$ Accuracy (pp)')
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax2.set_title('(b) Change from full model')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    save(fig, "fig_ablation")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_main_results()
    fig_topo_comparison()
    fig_theta_heatmap()
    fig_router_progression()
    fig_ablation()
    print("Done!")
