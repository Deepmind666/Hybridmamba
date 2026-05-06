#!/usr/bin/env python3
from __future__ import annotations

from matplotlib import pyplot as plt

BACKGROUND = "#ffffff"
PANEL = "#ffffff"
GRID = "#d9d9d9"
TEXT = "#111111"
MUTED = "#5f5f5f"

MODEL_COLORS = {
    "TinyViM_B": "#c8beb2",
    "HybridMamba-Base_B": "#d8adb1",
    "HybridMambaDet_B": "#8b5b73",
}

RANK_COLORS = {
    1: "#f4e3a1",
    2: "#eddcc9",
    3: "#efd1d8",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "font.family": ["DejaVu Sans"],
            "axes.edgecolor": "#111111",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "grid.linewidth": 0.55,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.dpi": 180,
        }
    )


def metric_to_percent(value: float) -> float:
    return value * 100.0


def fmt_percent(value: float, digits: int = 1) -> str:
    return f"{metric_to_percent(value):.{digits}f}"
