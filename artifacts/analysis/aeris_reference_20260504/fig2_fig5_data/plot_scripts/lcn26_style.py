"""Shared publication style for LCN26 figures.

The figures follow a conservative network-systems style: white background,
black axes, light dashed grids, one high-contrast color for AERIS, muted
baselines, and at most one secondary accent per plot.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


COLUMN_WIDTH_IN = 3.5
TEXT_WIDTH_IN = 7.16

PALETTE = {
    "AERIS": "#5A5A5A",
    "AERIS_dark": "#333333",
    "AERIS_band": "#5A5A5A",
    "PEGASIS": "#36A657",
    "RPL-MRHOF": "#FF7F0E",
    "RPL": "#FF7F0E",
    "CTP": "#4D4D4D",
    "LEACH": "#2D83BD",
    "HEED": "#C6373D",
    "TEEN": "#D15B9A",
    "classical": "#BDBDBD",
    "collection": "#4D4D4D",
    "GW": "#36A657",
    "CAS": "#2D83BD",
    "CHscore": "#9E9E9E",
    "cost": "#D62728",
    "secondary": "#2CA02C",
    "axis": "#111111",
    "muted": "#555555",
    "grid": "#CFCFCF",
    "panel_stress": "#FFFFFF",
    "panel_default": "#FFFFFF",
    "zero_line": "#111111",
    "neutral_band": "#E6E6E6",
}

MARKERS = {
    "AERIS": "o",
    "PEGASIS": "s",
    "RPL-MRHOF": "^",
    "RPL": "^",
    "CTP": "D",
    "LEACH": "v",
    "HEED": "P",
    "TEEN": "X",
}

LINEWIDTHS = {
    "AERIS": 1.85,
    "default": 0.95,
}

ALPHAS = {
    "AERIS": 1.0,
    "default": 0.78,
}


def apply_lcn26_style() -> None:
    plt.style.use("default")
    mpl.rcParams.update(
        {
            # TrueType embedding avoids Type 3 text in generated PDF figures.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 8.0,
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 320,
            "axes.edgecolor": PALETTE["axis"],
            "xtick.color": PALETTE["axis"],
            "ytick.color": PALETTE["axis"],
            "text.color": PALETTE["axis"],
            "axes.labelcolor": PALETTE["axis"],
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.95,
            "grid.linestyle": "--",
            "axes.grid.axis": "y",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.7,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 0.9,
            "legend.borderaxespad": 0.05,
        }
    )


def style_protocol_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, color=PALETTE["grid"])
    ax.tick_params(direction="out", length=2.4, width=0.6)


def linewidth(proto: str) -> float:
    return LINEWIDTHS["AERIS"] if proto == "AERIS" else LINEWIDTHS["default"]


def alpha(proto: str) -> float:
    return ALPHAS["AERIS"] if proto == "AERIS" else ALPHAS["default"]


def color(proto: str) -> str:
    return PALETTE.get(proto, PALETTE["muted"])


def marker(proto: str) -> str:
    return MARKERS.get(proto, "o")
