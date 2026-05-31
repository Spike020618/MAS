"""IEEE double-column publication figure styling."""
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np

# IEEE standard: 3.5" single column, 7.16" double column
SINGLE_COL = 3.5
DOUBLE_COL = 7.16
ASPECT = 0.75

# Color palette: distinguishable in grayscale print
COLORS = {
    "primary": "#2166ac",      # blue
    "secondary": "#b2182b",    # red
    "tertiary": "#1b7837",     # green
    "quaternary": "#762a83",   # purple
    "quinary": "#e08214",      # orange
    "gray": "#636363",
}
PALETTE = ["#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214"]
HATCHES = ["///", "\\\\\\", "xxx", "...", "+++"]


def ieee_style():
    """Apply IEEE publication figure style globally."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STSong"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "axes.unicode_minus": False,
        "axes.prop_cycle": cycler("color", PALETTE),
        "figure.constrained_layout.use": True,
    })


def significance_marker(ax, x1, x2, y, p_value):
    """Add significance bracket between two bars."""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "n.s."
    dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], lw=0.6, c="k")
    ax.text((x1 + x2) / 2, y + dy, stars, ha="center", va="bottom", fontsize=7)


def add_subfig_label(ax, label, x=-0.12, y=1.05):
    """Add (a), (b), (c) subfigure label."""
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="right")
