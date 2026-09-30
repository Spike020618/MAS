"""Generate publication-style architecture figures for the thesis and slides."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle


OUT = Path(__file__).resolve().parent

NAVY = "#17324D"
BLUE = "#3E6F9E"
BLUE_SOFT = "#EAF1F7"
TEAL = "#2D7F78"
TEAL_SOFT = "#E8F3F1"
ORANGE = "#C76D2A"
ORANGE_SOFT = "#F8EEE6"
GRAY = "#5D6975"
GRAY_SOFT = "#F3F5F7"
LINE = "#AAB4BE"
WHITE = "#FFFFFF"
PURPLE, PURPLE_SOFT = "#7557A5", "#F0EBF8"
CORAL, CORAL_SOFT = "#D95D55", "#FBECE9"
GREEN, GREEN_SOFT = "#5E963B", "#EDF6E8"


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def box(ax, x, y, w, h, title, lines=(), *, fc=WHITE, ec=LINE, color=NAVY,
        lw=1.2, radius=0.02, title_size=10, body_size=8.2, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.075 * h, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=color, zorder=zorder + 1)
    if lines:
        body = lines if isinstance(lines, str) else "\n".join(lines)
        ax.text(x + w / 2, y + h * 0.48, body, ha="center", va="center",
                fontsize=body_size, color=GRAY, linespacing=1.45, zorder=zorder + 1)
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=BLUE, lw=1.6, style="-|>", dashed=False,
          rad=0.0, zorder=1):
    p = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", zorder=zorder,
    )
    ax.add_patch(p)
    return p


def stage_label(ax, x, text, number, color):
    ax.text(x, 0.885, number, ha="center", va="center", fontsize=8, color=WHITE,
            bbox=dict(boxstyle="circle,pad=0.32", facecolor=color, edgecolor="none"))
    ax.text(x + 0.025, 0.885, text, ha="left", va="center", fontsize=9,
            fontweight="bold", color=color)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def research_framework():
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.985, "AEIC-MD: observable effort before aggregation",
            ha="left", va="top", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(0.03, 0.945, "The leader commits first; each follower chooses an action that selects a real execution path before settlement.",
            ha="left", va="top", fontsize=9, color=GRAY)

    stage_label(ax, 0.055, "Leader commitment", "1", BLUE)
    stage_label(ax, 0.295, "Follower response", "2", TEAL)
    stage_label(ax, 0.555, "Enforced execution", "3", ORANGE)
    stage_label(ax, 0.805, "Settlement + decision", "4", NAVY)

    box(ax, 0.025, 0.52, 0.20, 0.27, "Public mechanism offer",
        ("credit rule gₘ", "visible action costs cₘ(a)", "information policy + budget"),
        fc=BLUE_SOFT, ec=BLUE, color=BLUE)
    box(ax, 0.275, 0.52, 0.20, 0.27, "Action chosen before solving",
        ("LOW EFFORT", "AEIC EFFORT", "VERIFY", "DEFER / IMITATE"),
        fc=TEAL_SOFT, ec=TEAL, color=TEAL)
    box(ax, 0.525, 0.52, 0.20, 0.27, "Resource-tiered executor",
        ("action-specific prompt + model", "A  Assumption   E  Evidence", "I  Inference      C  Conclusion"),
        fc=ORANGE_SOFT, ec=ORANGE, color=ORANGE)
    box(ax, 0.775, 0.52, 0.20, 0.27, "Auditable controller",
        ("credit − cost → utility", "reputation update", "round-3 majority decision"),
        fc=GRAY_SOFT, ec=NAVY, color=NAVY)
    arrow(ax, 0.225, 0.655, 0.275, 0.655, color=BLUE)
    arrow(ax, 0.475, 0.655, 0.525, 0.655, color=TEAL)
    arrow(ax, 0.725, 0.655, 0.775, 0.655, color=ORANGE)
    arrow(ax, 0.875, 0.52, 0.125, 0.52, color=LINE, dashed=True, rad=0.16)

    ax.add_patch(Rectangle((0.215, 0.19), 0.64, 0.17, facecolor=WHITE,
                           edgecolor=LINE, linewidth=1.0))
    ax.text(0.235, 0.325, "Empirical evidence chain", fontsize=9.5,
            fontweight="bold", color=NAVY, va="top")
    evidence = [
        (0.24, "Formal test", "450 paired questions\n3 QA domains"),
        (0.40, "Follower response", "+56.47 pp high effort\n0.46% → 56.93%"),
        (0.58, "Decision effect", "+14.22 pp accuracy\nHolm p = 1.23×10⁻¹¹"),
        (0.75, "Boundary", "outcome-only: null\nheld-out GLM: null"),
    ]
    for x, t, s in evidence:
        ax.text(x, 0.275, t, fontsize=8.5, fontweight="bold", color=BLUE, ha="left")
        ax.text(x, 0.24, s, fontsize=7.5, color=GRAY, ha="left", va="top", linespacing=1.4)
    for x in (0.375, 0.555, 0.725):
        arrow(ax, x, 0.27, x + 0.022, 0.27, color=LINE, lw=1.1)

    ax.text(0.025, 0.075,
            "Scope: revealed-rule follower response on tested backends; no claim of universal incentive compatibility, equilibrium, or cross-backend transfer.",
            fontsize=8.2, color=GRAY, ha="left", va="center")
    save(fig, "fig_research_framework")


def system_architecture():
    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.985, "Movie search and recommendation: application-level validation of AEIC-MD",
            ha="left", va="top", fontsize=13.5, fontweight="bold", color=NAVY)
    ax.text(0.03, 0.945, "Deterministic retrieval grounds candidates; multi-agent reports are measured and controlled before evidence-aware ranking.",
            ha="left", va="top", fontsize=9, color=GRAY)

    layers = [
        (0.82, "INTERACTION", GRAY_SOFT, LINE),
        (0.65, "PLANNING", BLUE_SOFT, BLUE),
        (0.42, "CONSENSUS MECHANISM", TEAL_SOFT, TEAL),
        (0.19, "DECISION / AUDIT", ORANGE_SOFT, ORANGE),
    ]
    for y, label, fc, ec in layers:
        ax.add_patch(FancyBboxPatch((0.03, y), 0.94, 0.125 if y != 0.42 else 0.19,
                                   boxstyle="round,pad=0.008,rounding_size=0.015",
                                   facecolor=fc, edgecolor=ec, linewidth=1.0))
        ax.text(0.045, y + (0.105 if y != 0.42 else 0.17), label,
                fontsize=7.5, fontweight="bold", color=ec, va="top")

    box(ax, 0.09, 0.84, 0.18, 0.075, "User request",
        ("query · preferences · constraints"), fc=WHITE, title_size=8.8, body_size=7.2)
    box(ax, 0.41, 0.84, 0.18, 0.075, "HTTP / application API",
        ("typed schema · validation"), fc=WHITE, title_size=8.8, body_size=7.2)
    box(ax, 0.73, 0.84, 0.18, 0.075, "Top-K response",
        ("items · reasons · evidence"), fc=WHITE, title_size=8.8, body_size=7.2)
    arrow(ax, 0.27, 0.878, 0.41, 0.878, color=BLUE)
    arrow(ax, 0.59, 0.878, 0.73, 0.878, color=BLUE)

    box(ax, 0.09, 0.665, 0.18, 0.075, "Request planner",
        ("risk · roles · budget · protocol"), fc=WHITE, ec=BLUE, color=BLUE,
        title_size=8.8, body_size=7.2)
    box(ax, 0.32, 0.665, 0.18, 0.075, "Catalog retrieval",
        ("grounded candidates · constraints"), fc=WHITE, ec=BLUE, color=BLUE,
        title_size=8.8, body_size=7.2)
    box(ax, 0.55, 0.665, 0.18, 0.075, "Deterministic verifier",
        ("entity · metadata · hard rules"), fc=WHITE, ec=BLUE, color=BLUE,
        title_size=8.8, body_size=7.2)
    box(ax, 0.78, 0.665, 0.14, 0.075, "Fallback",
        ("timeout · invalid JSON"), fc=WHITE, ec=BLUE, color=BLUE,
        title_size=8.8, body_size=7.2)
    for x1, x2 in ((0.27, 0.32), (0.50, 0.55), (0.73, 0.78)):
        arrow(ax, x1, 0.703, x2, 0.703, color=BLUE)

    box(ax, 0.075, 0.475, 0.18, 0.09, "Preference agent",
        ("relevance · user intent"), fc=PURPLE_SOFT, ec=PURPLE, color=PURPLE,
        title_size=8.8, body_size=7.2)
    box(ax, 0.285, 0.475, 0.18, 0.09, "Diversity agent",
        ("coverage · redundancy"), fc=GREEN_SOFT, ec=GREEN, color=GREEN,
        title_size=8.8, body_size=7.2)
    box(ax, 0.495, 0.475, 0.18, 0.09, "Critic agent",
        ("conflict · unsupported claim"), fc=CORAL_SOFT, ec=CORAL, color=CORAL,
        title_size=8.8, body_size=7.2)
    box(ax, 0.715, 0.455, 0.215, 0.13, "AEIC consensus controller",
        ("report parsing + behavior metrics", "aggregation / intervention / routing", "evidence + reputation settlement"),
        fc=WHITE, ec=TEAL, color=TEAL, title_size=9.2, body_size=7.2)
    for x in (0.255, 0.465, 0.675):
        arrow(ax, x, 0.52, 0.715, 0.52, color=TEAL, lw=1.3)

    # Role badges and explicit AEIC packets make the application mapping visible.
    for x, mark, color in ((0.095, "P", PURPLE), (0.305, "D", GREEN), (0.515, "C", CORAL)):
        ax.add_patch(Circle((x, 0.548), 0.014, facecolor=color, edgecolor=WHITE,
                            linewidth=1.0, zorder=4))
        ax.text(x, 0.548, mark, ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=WHITE, zorder=5)
    for x in (0.225, 0.435, 0.645):
        ax.text(x, 0.535, "{A,E,I,C}", ha="center", va="bottom", fontsize=5.8,
                fontweight="bold", color=TEAL,
                bbox=dict(boxstyle="round,pad=.18", facecolor=TEAL_SOFT,
                          edgecolor=TEAL, linewidth=.6))

    ax.text(0.50, 0.625, "Research mechanism implemented here",
            ha="center", va="center", fontsize=8, fontweight="bold", color=TEAL)
    arrow(ax, 0.18, 0.665, 0.16, 0.565, color=LINE, dashed=True)
    arrow(ax, 0.41, 0.665, 0.375, 0.565, color=LINE, dashed=True)
    arrow(ax, 0.64, 0.665, 0.585, 0.565, color=LINE, dashed=True)

    box(ax, 0.08, 0.205, 0.20, 0.075, "Evidence-aware reranker",
        ("constraint-safe consensus score"), fc=WHITE, ec=ORANGE, color=ORANGE,
        title_size=8.8, body_size=7.2)
    box(ax, 0.34, 0.205, 0.20, 0.075, "Decision package",
        ("Top-K · dissent · provenance"), fc=WHITE, ec=ORANGE, color=ORANGE,
        title_size=8.8, body_size=7.2)
    box(ax, 0.60, 0.205, 0.15, 0.075, "Trace store",
        ("plan · reports · costs"), fc=WHITE, ec=ORANGE, color=ORANGE,
        title_size=8.8, body_size=7.2)
    box(ax, 0.80, 0.205, 0.12, 0.075, "Metrics",
        ("quality · latency"), fc=WHITE, ec=ORANGE, color=ORANGE,
        title_size=8.8, body_size=7.2)
    arrow(ax, 0.825, 0.455, 0.18, 0.28, color=ORANGE, rad=0.1)
    arrow(ax, 0.28, 0.243, 0.34, 0.243, color=ORANGE)
    arrow(ax, 0.54, 0.243, 0.60, 0.243, color=ORANGE)
    arrow(ax, 0.75, 0.243, 0.80, 0.243, color=ORANGE)
    arrow(ax, 0.44, 0.205, 0.82, 0.915, color=LINE, dashed=True, rad=-0.18)

    # Grounding and observability are shared services, not hidden inside the LLM boxes.
    ax.add_patch(FancyBboxPatch((0.10, 0.115), 0.78, 0.045,
                               boxstyle="round,pad=.004,rounding_size=.012",
                               facecolor=WHITE, edgecolor=LINE, linewidth=.9))
    services = [(0.15, "Movie catalog", BLUE), (0.34, "Prompt / policy registry", PURPLE),
                (0.57, "Trace + provenance store", ORANGE), (0.78, "Metrics dashboard", TEAL)]
    for x, label, color in services:
        ax.add_patch(Circle((x, 0.137), .008, facecolor=color, edgecolor="none"))
        ax.text(x+.014, .137, label, ha="left", va="center", fontsize=6.3,
                fontweight="bold", color=color)

    ax.text(0.03, 0.065,
            "Validation endpoints: constraint satisfaction · invalid-candidate rate · evidence support · NDCG@K · calls/tokens · P95 latency · degradation success",
            fontsize=8.2, color=GRAY, ha="left", va="center")
    save(fig, "fig_system_architecture")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    research_framework()
    system_architecture()
    print(f"Wrote architecture figures to {OUT}")
