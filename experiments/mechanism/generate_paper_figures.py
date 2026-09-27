"""Generate the paper-facing figures directly from frozen AEIC-MD bundles.

The script deliberately reads only ``results/mechanism/frozen``.  It creates
PNG and PDF counterparts under ``results/mechanism/figures`` and a compact
machine-readable source table alongside them.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# Headless backend keeps the figure build reproducible on CI and macOS hosts.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results" / "mechanism" / "frozen"
OUT = ROOT / "results" / "mechanism" / "figures"

COLORS = {"no": "#8c8c8c", "outcome": "#d95f02", "static": "#1b9e77"}


def load(suite: str) -> dict:
    return json.loads((FROZEN / suite / "analysis.json").read_text(encoding="utf-8"))


def metric(conditions: dict, condition: str, field: str) -> float:
    return float(conditions[condition][field])


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def formal_figure(source: dict) -> None:
    conditions = load("formal_v10")["conditions"]
    keys = [
        ("No mechanism", "independent__no_mechanism@0.2", "no"),
        ("Outcome only", "independent__outcome_only@0.2", "outcome"),
        ("Static AEIC", "independent__static_aeic@0.2", "static"),
    ]
    fields = [("decision_quality", "Decision quality", (0.0, 1.0)),
              ("high_effort_choice_rate", "High-effort choice rate", (0.0, 0.7)),
              ("mean_tokens_per_agent_round", "Tokens / agent-round", (0.0, 950.0))]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, (field, title, ylim) in zip(axes, fields):
        values = [metric(conditions, key, field) for _, key, _ in keys]
        bars = ax.bar(range(3), values, color=[COLORS[color] for _, _, color in keys])
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(3), [label for label, _, _ in keys], rotation=18, ha="right")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + (ylim[1] - ylim[0]) * .025,
                    f"{value:.1%}" if field != "mean_tokens_per_agent_round" else f"{value:.0f}",
                    ha="center", va="bottom", fontsize=9)
    source["formal"] = {label: {field: metric(conditions, key, field) for field, _, _ in fields}
                        for label, key, _ in keys}
    save(fig, "fig1_formal_mechanism")


def boundary_figure(source: dict) -> None:
    report_path = OUT / "boundary_curves.json"
    if not report_path.exists():
        raise FileNotFoundError(
            "Generate boundary_curves.json first; see docs/frozen_results_storyboard.md"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    diagnostics = report["diagnostics"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    for dataset, values in diagnostics.items():
        reward = values["reward_response"]
        cost = values["cost_response"]
        axes[0].plot([x for x, _ in reward], [y for _, y in reward], marker="o", label=dataset)
        axes[1].plot([x for x, _ in cost], [y for _, y in cost], marker="o", label=dataset)
    axes[0].set_xlabel("Reward scale")
    axes[0].set_ylabel("High-effort choice rate")
    axes[0].set_title("Reward response")
    axes[1].set_xlabel("Effort-cost multiplier")
    axes[1].set_title("Cost response")
    for ax in axes:
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=.25)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    source["boundary"] = diagnostics
    save(fig, "fig2_incentive_response")


def information_figure(source: dict) -> None:
    conditions = load("information_v11_dev")["conditions"]
    behaviors = [("Evidence", "evidence_update"), ("Social", "social_only"),
                 ("Copy pressure", "copy_pressure"), ("Counterevidence", "counterevidence")]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1))
    for ax, field, title in zip(axes, ("decision_quality", "high_effort_choice_rate"),
                                ("Decision quality", "High-effort choice rate")):
        xs = list(range(len(behaviors)))
        width = .35
        no = [metric(conditions, f"{prefix}__no_mechanism@0.2", field) for _, prefix in behaviors]
        static = [metric(conditions, f"{prefix}__static_aeic@0.2", field) for _, prefix in behaviors]
        ax.bar([x - width / 2 for x in xs], no, width, label="No mechanism", color=COLORS["no"])
        ax.bar([x + width / 2 for x in xs], static, width, label="Static AEIC", color=COLORS["static"])
        ax.set_xticks(xs, [label for label, _ in behaviors], rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.0 if field == "decision_quality" else .75)
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, loc="lower left")
    source["information"] = {label: {
        "no_quality": metric(conditions, f"{prefix}__no_mechanism@0.2", "decision_quality"),
        "static_quality": metric(conditions, f"{prefix}__static_aeic@0.2", "decision_quality"),
        "no_effort": metric(conditions, f"{prefix}__no_mechanism@0.2", "high_effort_choice_rate"),
        "static_effort": metric(conditions, f"{prefix}__static_aeic@0.2", "high_effort_choice_rate"),
    } for label, prefix in behaviors}
    save(fig, "fig3_information_conditions")


def robustness_figure(source: dict) -> None:
    suites = [
        ("Signal ablation", "signal_ablation_v11_dev", ["AEIC", "EIC", "AIC", "AEC", "AEI"], "signals@"),
        ("History ablation", "history_ablation_v11_dev", ["full", "reset", "hidden", "shuffled"], "history@"),
        ("Panel size", "missing_agent_v11_dev", ["agents3", "agents4", "agents5"], "variant@"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    source["robustness"] = {}
    for ax, (title, suite, labels, marker) in zip(axes, suites):
        conditions = load(suite)["conditions"]
        selected = []
        for label in labels:
            if suite == "history_ablation_v11_dev" and label == "full":
                key = "independent__static_aeic@0.2"
            else:
                key = next(k for k in conditions if marker + label in k)
            selected.append((label, conditions[key]))
        values = [float(item["decision_quality"]) for _, item in selected]
        bars = ax.bar(range(len(labels)), values, color=COLORS["static"])
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(0.6, 1.0)
        ax.grid(axis="y", alpha=.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + .01, f"{value:.1%}", ha="center", fontsize=9)
        source["robustness"][title] = dict(zip(labels, values))
    axes[0].set_ylabel("Decision quality")
    save(fig, "fig4_ablations_robustness")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    source: dict = {"source": "results/mechanism/frozen/manifest.json"}
    formal_figure(source)
    boundary_figure(source)
    information_figure(source)
    robustness_figure(source)
    (OUT / "figure_source_table.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
