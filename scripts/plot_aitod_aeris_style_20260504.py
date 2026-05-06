from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\mamba")
SOURCE = ROOT / "artifacts" / "analysis" / "aitod_current_20260504_1021" / "aitod_cvpr_all_metrics.csv"
OUT = ROOT / "artifacts" / "analysis" / "aitod_aeris_style_20260504_1736"
RAW = OUT / "raw"


VAL_RE = re.compile(
    r"Epoch\(val\)\s+\[(?P<epoch>\d+)\]\[\d+/\d+\]\s+"
    r"coco/bbox_mAP:\s+(?P<ap>-?\d+\.\d+)\s+"
    r"coco/bbox_mAP_50:\s+(?P<ap50>-?\d+\.\d+)\s+"
    r"coco/bbox_mAP_75:\s+(?P<ap75>-?\d+\.\d+)\s+"
    r"coco/bbox_mAP_s:\s+(?P<aps>-?\d+\.\d+)\s+"
    r"coco/bbox_mAP_m:\s+(?P<apm>-?\d+\.\d+)"
)


DISPLAY = {
    "TinyViM-B + RetinaNet": "TinyViM-B",
    "HybridMamba-B + RetinaNet": "HybridMamba-B",
    "HybridMambaDet-B + RetinaNet": "HybridMambaDet-B",
    "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5": "HybridMambaDet-B alpha=0.5",
    "HybridMambaDet-B + RetinaNet, tiny-object protocol": "HybridMambaDet-B tiny-object",
}

AXIS_DISPLAY = {
    "TinyViM-B + RetinaNet": "TinyViM-B",
    "HybridMamba-B + RetinaNet": "HybridMamba-B",
    "HybridMambaDet-B + RetinaNet": "HybridMambaDet-B",
    "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5": "HybridMambaDet alpha=0.5",
    "HybridMambaDet-B + RetinaNet, tiny-object protocol": "HybridMambaDet tiny",
}


# Mirrors the user's AERIS LCN26 style pack: white background, dark proposed
# method, green/blue/red/pink baselines, dashed light grid, compact text.
PALETTE = {
    "HybridMambaDet-B + RetinaNet": "#5A5A5A",
    "TinyViM-B + RetinaNet": "#36A657",
    "HybridMamba-B + RetinaNet": "#2D83BD",
    "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5": "#C6373D",
    "HybridMambaDet-B + RetinaNet, tiny-object protocol": "#D15B9A",
    "axis": "#111111",
    "grid": "#CFCFCF",
    "muted": "#555555",
}

MARKERS = {
    "HybridMambaDet-B + RetinaNet": "o",
    "TinyViM-B + RetinaNet": "s",
    "HybridMamba-B + RetinaNet": "^",
    "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5": "D",
    "HybridMambaDet-B + RetinaNet, tiny-object protocol": "P",
}

METHOD_ORDER = [
    "TinyViM-B + RetinaNet",
    "HybridMambaDet-B + RetinaNet",
    "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5",
    "HybridMambaDet-B + RetinaNet, tiny-object protocol",
    "HybridMamba-B + RetinaNet",
]


def apply_aeris_style() -> None:
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
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
            "savefig.dpi": 420,
            "axes.edgecolor": PALETTE["axis"],
            "xtick.color": PALETTE["axis"],
            "ytick.color": PALETTE["axis"],
            "text.color": PALETTE["axis"],
            "axes.labelcolor": PALETTE["axis"],
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.95,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
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


def style_axes(ax: plt.Axes, ylabel: str, xlabel: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, color=PALETTE["grid"])
    ax.tick_params(direction="out", length=2.4, width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Validation epoch" if xlabel else "")


def parse_log(path: Path, method: str) -> pd.DataFrame:
    rows: list[dict] = []
    if not path.exists():
        return pd.DataFrame()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = VAL_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "run": str(path),
                "method": method,
                "epoch": int(match.group("epoch")),
                "AP": float(match.group("ap")),
                "AP50": float(match.group("ap50")),
                "AP75": float(match.group("ap75")),
                "AP_S": float(match.group("aps")),
                "AP_M": float(match.group("apm")),
            }
        )
    return pd.DataFrame(rows)


def load_data() -> pd.DataFrame:
    if SOURCE.exists():
        df = pd.read_csv(SOURCE)
    else:
        df = pd.DataFrame(columns=["run", "method", "epoch", "AP", "AP50", "AP75", "AP_S", "AP_M"])

    live = [
        parse_log(RAW / "local_tinyvim_b_train.log", "TinyViM-B + RetinaNet"),
        parse_log(
            RAW / "fat_tinyproto_train.log",
            "HybridMambaDet-B + RetinaNet, tiny-object protocol",
        ),
    ]
    df = pd.concat([df, *live], ignore_index=True)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df = (
        df.sort_values(["method", "epoch", "run"])
        .drop_duplicates(["method", "epoch"], keep="last")
        .sort_values(["method", "epoch"])
        .reset_index(drop=True)
    )
    return df


def method_lw(method: str) -> float:
    return 1.85 if "HybridMambaDet-B + RetinaNet" == method else 1.05


def method_alpha(method: str) -> float:
    return 1.0 if "HybridMambaDet" in method else 0.82


def save_all(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(df: pd.DataFrame) -> None:
    metrics = [("AP", "AP"), ("AP50", "AP50"), ("AP75", "AP75"), ("AP_S", "AP small")]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9), sharex=True)
    for idx, (ax, (metric, label)) in enumerate(zip(axes.flat, metrics)):
        ymax = max(0.11, float(df[metric].max()) * 1.16)
        for method in METHOD_ORDER:
            part = df[df["method"] == method].sort_values("epoch")
            if part.empty:
                continue
            linestyle = "--" if "tiny-object" in method else "-"
            ax.plot(
                part["epoch"],
                part[metric],
                color=PALETTE[method],
                lw=method_lw(method),
                marker=MARKERS[method],
                ms=3.4,
                mec=PALETTE[method],
                mew=0.45,
                linestyle=linestyle,
                alpha=method_alpha(method),
                label=DISPLAY[method],
            )
            best = part.loc[part[metric].idxmax()]
            ax.scatter(
                [best["epoch"]],
                [best[metric]],
                s=38,
                facecolor="white",
                edgecolor=PALETTE[method],
                linewidth=1.05,
                zorder=5,
            )
        ax.set_ylim(0, ymax)
        ax.set_xlim(0.7, 18.7)
        ax.set_xticks([1, 4, 8, 12, 16, 18])
        style_axes(ax, label, xlabel=idx >= 2)
        ax.set_title(label, pad=3.0)
        ax.text(
            0.01,
            0.96,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=8.8,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.09, top=0.875, wspace=0.20, hspace=0.35)
    save_all(fig, "figure_aeris_validation_trajectory")


def best_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        part = df[df["method"] == method]
        if part.empty:
            continue
        best = part.loc[part["AP"].idxmax()].copy()
        latest = part.sort_values("epoch").iloc[-1]
        rows.append(
            {
                "method": method,
                "display": DISPLAY[method],
                "axis_display": AXIS_DISPLAY[method],
                "epoch_best": int(best["epoch"]),
                "AP_best": float(best["AP"]),
                "AP50_best": float(best["AP50"]),
                "AP75_best": float(best["AP75"]),
                "AP_S_best": float(best["AP_S"]),
                "epoch_latest": int(latest["epoch"]),
                "AP_latest": float(latest["AP"]),
                "ongoing": "tiny-object" in method,
            }
        )
    return pd.DataFrame(rows)


def plot_best_and_gap(df: pd.DataFrame) -> None:
    summary = best_summary(df)
    ref = float(summary.loc[summary["method"] == "TinyViM-B + RetinaNet", "AP_best"].iloc[0])
    summary["gap_vs_tinyvim"] = summary["AP_best"] - ref
    summary.to_csv(OUT / "aeris_style_best_summary.csv", index=False)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 3.15),
        gridspec_kw={"width_ratios": [1.10, 1.06]},
        constrained_layout=True,
    )
    ax = axes[0]
    x = np.arange(len(summary))
    width = 0.18
    metric_cols = [("AP_best", "AP"), ("AP50_best", "AP50"), ("AP75_best", "AP75"), ("AP_S_best", "AP small")]
    metric_colors = ["#5A5A5A", "#36A657", "#2D83BD", "#C6373D"]
    for i, (col, label) in enumerate(metric_cols):
        bars = ax.bar(
            x + (i - 1.5) * width,
            summary[col],
            width=width,
            color=metric_colors[i],
            edgecolor="#111111",
            linewidth=0.35,
            label=label,
        )
        for bar, ongoing in zip(bars, summary["ongoing"]):
            if ongoing:
                bar.set_hatch("///")
                bar.set_alpha(0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["axis_display"], rotation=16, ha="right")
    ax.set_ylabel("Best validation score")
    ax.set_ylim(0, max(0.27, summary["AP50_best"].max() * 1.12))
    style_axes(ax, "Best validation score")
    ax.legend(loc="upper left", ncol=4, bbox_to_anchor=(-0.01, 1.13))
    ax.text(0.01, 0.96, "a", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=8.8)

    ax2 = axes[1]
    ypos = np.arange(len(summary))
    bars = ax2.barh(
        ypos,
        summary["gap_vs_tinyvim"],
        color=[PALETTE[m] for m in summary["method"]],
        edgecolor="#111111",
        linewidth=0.35,
    )
    for bar, ongoing in zip(bars, summary["ongoing"]):
        if ongoing:
            bar.set_hatch("///")
            bar.set_alpha(0.72)
    ax2.axvline(0, color="#111111", lw=0.75)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(summary["axis_display"])
    ax2.set_xlabel("Best AP difference vs TinyViM-B")
    ax2.grid(axis="x", linestyle="--", linewidth=0.5, color=PALETTE["grid"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="y", labelsize=6.4, pad=1.5)
    for idx, val in enumerate(summary["gap_vs_tinyvim"]):
        ax2.text(
            -0.0022 if val < 0 else 0.0018,
            idx,
            f"{val:+.3f}",
            va="center",
            ha="right" if val < 0 else "left",
            fontsize=7.0,
            color=PALETTE["axis"],
        )
    ax2.text(0.01, 0.96, "b", transform=ax2.transAxes, ha="left", va="top", fontweight="bold", fontsize=8.8)
    save_all(fig, "figure_aeris_best_gap")


def plot_ap50_diagnostic(df: pd.DataFrame) -> None:
    summary = best_summary(df)
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    x = np.arange(len(summary))
    vals = summary["AP50_best"] - summary["AP_best"]
    bars = ax.bar(
        x,
        vals,
        color=[PALETTE[m] for m in summary["method"]],
        edgecolor="#111111",
        linewidth=0.35,
    )
    for bar, ongoing in zip(bars, summary["ongoing"]):
        if ongoing:
            bar.set_hatch("///")
            bar.set_alpha(0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["axis_display"], rotation=20, ha="right")
    style_axes(ax, "AP50 - AP")
    ax.set_title("AP50-to-AP gap", loc="left", pad=3)
    for i, value in enumerate(vals):
        ax.text(i, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=6.8)
    save_all(fig, "figure_aeris_ap50_gap_diagnostic")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    apply_aeris_style()
    df = load_data()
    df.to_csv(OUT / "aeris_style_all_metrics.csv", index=False)
    (OUT / "aeris_style_all_metrics.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    plot_trajectory(df)
    plot_best_and_gap(df)
    plot_ap50_diagnostic(df)


if __name__ == "__main__":
    main()
