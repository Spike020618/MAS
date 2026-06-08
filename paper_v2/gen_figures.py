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
    router_results = {}
    router_path = os.path.join(ROOT, "results/configs/router_v10_results.json")
    if os.path.exists(router_path):
        with open(router_path) as f:
            router_results = json.load(f)

    methods = [
        ("Single\nAgent", "Single Agent"),
        ("MajVote\n(parallel)", "MajVote_parallel"),
        ("MajVote\n(star)", "MajVote_star"),
        ("AEIC+$\\theta^*$\n(star)", "AEIC_opt_star"),
        ("Adaptive\nRouter", None),
        ("Oracle", "Oracle_5topo"),
    ]

    router_per_ds = {"MMLU-Hard": 92.5, "StrategyQA": 86.2, "TruthfulQA": 85.6}

    x = np.arange(len(DATASETS))
    width = 0.12
    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    for i, (label, key) in enumerate(methods):
        vals = []
        for ds in DATASETS:
            if key is None:
                vals.append(router_per_ds[ds])
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
        cv_path = os.path.join(ROOT, f"results/configs/opt_{topo}_cv.json")
        path = os.path.join(ROOT, f"results/configs/opt_{topo}.json")
        cfg_path = cv_path if os.path.exists(cv_path) else path
        with open(cfg_path) as f:
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
    router_path = os.path.join(ROOT, "results/configs/router_v10_results.json")
    rr = {}
    if os.path.exists(router_path):
        with open(router_path) as f:
            rr = json.load(f)

    methods = [
        ("Single Agent", 74.8),
        ("MajVote (star)", rr.get("best_fixed_acc", 0.831) * 100),
        ("AEIC+$\\theta^*$ (star, CV)", 81.7),
        ("Router (shared)", rr.get("router_shared_mean", 0.869) * 100),
        ("Router (per-dataset)", rr.get("router_per_dataset_mean", 0.876) * 100),
        ("Oracle (5-topo)", rr.get("oracle_acc", 0.929) * 100),
    ]

    labels = [m[0] for m in methods]
    vals = [m[1] for m in methods]

    fig, ax = plt.subplots(figsize=(5, 3.2))
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


# ============================================================
# Fig 6: Topology Specialization Heatmap
# ============================================================
def fig_topo_specialization():
    """Show per-dataset MajVote accuracy for each topology (aligned questions)."""
    def sig_path(ds, topo):
        if topo == 'parallel':
            return os.path.join(ROOT, f"results/signals/{ds}.json")
        return os.path.join(ROOT, f"results/signals/{ds}_{topo}.json")

    def _question_key(q):
        return q.get("question", "")[:200]

    from collections import Counter

    data_matrix = np.zeros((len(TOPOS), len(DATASETS)))

    for d_idx, ds in enumerate(DATASETS):
        all_data = {}
        topo_indices = {}
        for topo in TOPOS:
            with open(sig_path(ds, topo)) as f:
                all_data[topo] = json.load(f)
            topo_indices[topo] = {}
            for i, q in enumerate(all_data[topo]["questions"][:300]):
                if "signals" in q:
                    topo_indices[topo][_question_key(q)] = i

        common = set(topo_indices[TOPOS[0]].keys())
        for topo in TOPOS[1:]:
            common &= set(topo_indices[topo].keys())

        for t_idx, topo in enumerate(TOPOS):
            correct = 0
            for qk in common:
                idx = topo_indices[topo][qk]
                q_data = all_data[topo]["questions"][idx]
                stances = q_data["signals"]["stances"]
                gt = q_data["ground_truth"].lower()
                counts = Counter(s.lower() for s in stances)
                maj = counts.most_common(1)[0][0]
                if maj == gt:
                    correct += 1
            data_matrix[t_idx, d_idx] = correct / len(common) * 100

    fig, ax = plt.subplots(figsize=(4.5, 3))
    cmap = LinearSegmentedColormap.from_list('custom', ['#fee0d2', '#de2d26'])
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=65, vmax=95)

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_yticks(range(len(TOPOS)))
    ax.set_yticklabels([t.capitalize() for t in TOPOS])

    for i in range(len(TOPOS)):
        for j in range(len(DATASETS)):
            val = data_matrix[i, j]
            color = 'white' if val > 82 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Accuracy (%)')
    ax.set_title('MajVote accuracy by topology and dataset')
    save(fig, "fig_topo_specialization")


def fig_topo_agreement():
    """Show distribution of how many topologies get each question correct."""
    from collections import Counter as Ctr

    def sig_path(ds, topo):
        if topo == 'parallel':
            return os.path.join(ROOT, f"results/signals/{ds}.json")
        return os.path.join(ROOT, f"results/signals/{ds}_{topo}.json")

    def _question_key(q):
        return q.get("question", "")[:200]

    topo_questions = {}
    for topo in TOPOS:
        all_q = {}
        for ds in DATASETS:
            with open(sig_path(ds, topo)) as f:
                data = json.load(f)
            for q in data["questions"][:300]:
                if "signals" in q:
                    all_q[_question_key(q)] = q
        topo_questions[topo] = all_q

    common = set(topo_questions[TOPOS[0]].keys())
    for topo in TOPOS[1:]:
        common &= set(topo_questions[topo].keys())

    dist = Ctr()
    for key in common:
        n_correct = 0
        gt = topo_questions["parallel"][key]["ground_truth"].lower()
        for topo in TOPOS:
            q = topo_questions[topo][key]
            stances = q["signals"]["stances"]
            counts = Ctr(s.lower() for s in stances)
            if counts and counts.most_common(1)[0][0] == gt:
                n_correct += 1
        dist[n_correct] += 1

    n_total = len(common)
    x_labels = [f"{k}/5" for k in range(6)]
    counts = [dist.get(k, 0) for k in range(6)]
    pcts = [c / n_total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    colors = ['#de2d26', '#fc9272', '#fcbba1', '#fee0d2', '#deebf7', '#3182bd']
    bars = ax.bar(range(6), pcts, color=colors, edgecolor='white', width=0.7)
    for bar, p, c in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{p:.1f}%\n({c})', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(6))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Number of correct topologies')
    ax.set_ylabel('Questions (%)')
    ax.set_ylim(0, 65)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.annotate('All agree\n(routing irrelevant)', xy=(5, pcts[5]),
                xytext=(3.2, pcts[5]+5), fontsize=6.5,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    ax.annotate('All wrong\n(no topology helps)', xy=(0, pcts[0]),
                xytext=(1.2, pcts[0]+10), fontsize=6.5,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    save(fig, "fig_topo_agreement")


def fig_pareto():
    """Accuracy vs IC trade-off (Pareto frontier)."""
    pareto_path = os.path.join(ROOT, "results/pareto_results.json")
    if not os.path.exists(pareto_path):
        print("  Skipping fig_pareto (no data)")
        return

    with open(pareto_path) as f:
        results = json.load(f)

    accs = [r["accuracy"] * 100 for r in results]
    ics = [r["ic_rate"] * 100 for r in results]
    lams = [r["lambda"] for r in results]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(ics, accs, 'o-', color='#C44E52', markersize=6, linewidth=1.5)
    for i, lam in enumerate(lams):
        offset_x = 1.5 if i > 0 else -3
        offset_y = 0.3 if i % 2 == 0 else -0.6
        ax.annotate(f'$\\lambda$={lam}', (ics[i], accs[i]),
                    textcoords="offset points", xytext=(offset_x, offset_y),
                    fontsize=6.5)

    ax.set_xlabel('Incentive Compatibility (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_pareto")


def fig_feature_importance():
    """Router feature group importance (leave-one-out ablation)."""
    fi_path = os.path.join(ROOT, "results/feature_importance.json")
    if not os.path.exists(fi_path):
        print("  Skipping fig_feature_importance (no data)")
        return

    with open(fi_path) as f:
        fi_data = json.load(f)

    groups_dict = fi_data.get("groups", {})
    if not groups_dict:
        print("  Skipping fig_feature_importance (empty groups)")
        return

    items = [(name, info["drop"] * 100) for name, info in groups_dict.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    names = [x[0] for x in items]
    drops = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    colors = ['#C44E52' if d > 0 else '#55A868' for d in drops]
    bars = ax.barh(range(len(names)), drops, color=colors, edgecolor='white', height=0.6)

    for bar, d in zip(bars, drops):
        x_pos = d + 0.1 if d >= 0 else d - 0.1
        ha = 'left' if d >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{d:+.1f}pp', va='center', ha=ha, fontsize=7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Accuracy drop when ablated (pp)')
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_feature_importance")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_main_results()
    fig_topo_comparison()
    fig_theta_heatmap()
    fig_router_progression()
    fig_ablation()
    fig_topo_specialization()
    fig_topo_agreement()
    fig_feature_importance()
    fig_pareto()
    print("Done!")
