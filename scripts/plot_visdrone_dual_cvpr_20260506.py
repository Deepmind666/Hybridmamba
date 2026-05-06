#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\mamba")
ANALYSIS_DIR = ROOT / "artifacts" / "analysis" / "dual_current_visdrone_20260506_0800"
CURVES_CSV = ANALYSIS_DIR / "dual_current_validation_curves.csv"
SUMMARY_CSV = ANALYSIS_DIR / "dual_current_summary.csv"
OUT_ROOT = ROOT / "artifacts" / "figures"


METHOD_ORDER = [
    "TinyViM-B + RetinaNet",
    "MobileMamba-B1 + RetinaNet",
]

COLORS = {
    "TinyViM-B + RetinaNet": "#228833",
    "MobileMamba-B1 + RetinaNet": "#4477AA",
    "Gap": "#CC3311",
    "Axis": "#111111",
    "Grid": "#D8D8D8",
    "Muted": "#666666",
    "FillTiny": "#BDE5C8",
    "FillMobile": "#B9D4F0",
}

MARKERS = {
    "TinyViM-B + RetinaNet": "o",
    "MobileMamba-B1 + RetinaNet": "s",
}

METRICS = [
    ("AP", "AP"),
    ("AP50", "AP50"),
    ("AP75", "AP75"),
    ("AP_S", "AP small"),
    ("AP_M", "AP medium"),
    ("AP_L", "AP large"),
]


def setup_style() -> None:
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
            "axes.labelsize": 8.7,
            "axes.titlesize": 9.4,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.dpi": 600,
            "axes.edgecolor": COLORS["Axis"],
            "xtick.color": COLORS["Axis"],
            "ytick.color": COLORS["Axis"],
            "text.color": COLORS["Axis"],
            "axes.labelcolor": COLORS["Axis"],
            "grid.color": COLORS["Grid"],
            "grid.linewidth": 0.62,
            "grid.alpha": 1.0,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.82,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
        }
    )


def style_axes(ax: plt.Axes, *, xlabel: bool = False, ylabel: str = "AP points") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["Axis"])
    ax.spines["bottom"].set_color(COLORS["Axis"])
    ax.grid(axis="y", which="major")
    ax.tick_params(direction="out", length=3.0, width=0.75, pad=2.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Validation epoch" if xlabel else "")


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        -0.13,
        1.05,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
    )


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = pd.read_csv(CURVES_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    curves = curves[curves["method"].isin(METHOD_ORDER)].copy()
    summary = summary[summary["method"].isin(METHOD_ORDER)].copy()
    curves["epoch"] = curves["epoch"].astype(int)
    for col, _ in METRICS:
        curves[col] = pd.to_numeric(curves[col], errors="coerce")
    numeric_summary = [
        "n_val_epochs",
        "first_epoch",
        "latest_epoch",
        "latest_AP",
        "latest_AP50",
        "latest_AP75",
        "best_epoch",
        "best_AP",
        "best_AP50",
        "best_AP75",
        "best_AP_S",
        "best_AP_M",
        "best_AP_L",
    ]
    for col in numeric_summary:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return curves, summary


def metric_best(curves: pd.DataFrame, method: str, metric: str) -> pd.Series:
    part = curves[curves["method"] == method].sort_values("epoch")
    return part.loc[part[metric].idxmax()]


def latest(curves: pd.DataFrame, method: str) -> pd.Series:
    part = curves[curves["method"] == method].sort_values("epoch")
    return part.iloc[-1]


def plot_validation_curves(curves: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.9, 6.85), sharex=True)
    letters = list("abcdef")

    for idx, (ax, (metric, title)) in enumerate(zip(axes.flat, METRICS)):
        ymax = float(curves[metric].max())
        upper = max(1.0, ymax * 1.22 + 0.25)
        ax.set_ylim(0, upper)
        ax.set_xlim(1, 100)
        ax.set_xticks([1, 20, 40, 60, 80, 100])

        for method in METHOD_ORDER:
            part = curves[curves["method"] == method].sort_values("epoch")
            if part.empty:
                continue
            color = COLORS[method]
            ax.plot(
                part["epoch"],
                part[metric],
                color=color,
                lw=2.2,
                marker=MARKERS[method],
                markevery=max(1, len(part) // 8),
                ms=4.1,
                mec=color,
                mew=0.85,
                mfc="white",
                alpha=0.98,
                label=method,
                zorder=3,
            )
            best = metric_best(curves, method, metric)
            end = latest(curves, method)
            ax.scatter(
                [best["epoch"]],
                [best[metric]],
                s=56,
                facecolor="white",
                edgecolor=color,
                linewidth=1.65,
                zorder=5,
            )
            ax.scatter(
                [end["epoch"]],
                [end[metric]],
                s=31,
                marker=MARKERS[method],
                facecolor=color,
                edgecolor=color,
                linewidth=0.7,
                zorder=4,
            )

        style_axes(ax, xlabel=idx >= 3)
        ax.set_title(title, loc="left", pad=6.5)
        panel_label(ax, letters[idx])

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.012),
    )
    fig.suptitle(
        "TinyViM-B maintains a stronger VisDrone2019 validation trajectory under the same RetinaNet protocol",
        y=1.065,
        fontsize=10.6,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.084, top=0.895, wspace=0.255, hspace=0.31)
    save_all(fig, out_dir, "figure_1_visdrone_validation_trajectory")


def best_summary_from_curves(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        row: dict[str, object] = {"method": method}
        for metric, _ in METRICS:
            best = metric_best(curves, method, metric)
            row[f"best_{metric}"] = float(best[metric])
            row[f"best_{metric}_epoch"] = int(best["epoch"])
        end = latest(curves, method)
        for metric, _ in METRICS:
            row[f"latest_{metric}"] = float(end[metric])
        row["latest_epoch"] = int(end["epoch"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_best_metric_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.45, 3.95))
    x = np.arange(len(METRICS))
    width = 0.34

    tiny = summary[summary["method"] == METHOD_ORDER[0]].iloc[0]
    mobile = summary[summary["method"] == METHOD_ORDER[1]].iloc[0]
    summary_cols = ["best_AP", "best_AP50", "best_AP75", "best_AP_S", "best_AP_M", "best_AP_L"]
    tiny_vals = np.array([tiny[col] for col in summary_cols], dtype=float)
    mobile_vals = np.array([mobile[col] for col in summary_cols], dtype=float)

    bars_tiny = ax.bar(
        x - width / 2,
        tiny_vals,
        width=width,
        color=COLORS["TinyViM-B + RetinaNet"],
        edgecolor=COLORS["Axis"],
        linewidth=0.55,
        label=METHOD_ORDER[0],
        zorder=3,
    )
    bars_mobile = ax.bar(
        x + width / 2,
        mobile_vals,
        width=width,
        color=COLORS["MobileMamba-B1 + RetinaNet"],
        edgecolor=COLORS["Axis"],
        linewidth=0.55,
        label=METHOD_ORDER[1],
        zorder=3,
    )

    for bars in (bars_tiny, bars_mobile):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.42,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
            )

    for i, gap in enumerate(tiny_vals - mobile_vals):
        ymax = max(tiny_vals[i], mobile_vals[i])
        ax.plot([i - width / 2, i + width / 2], [ymax + 2.0, ymax + 2.0], color=COLORS["Gap"], lw=0.9)
        gap_text = f"{gap:+.1f}"
        ax.text(
            i,
            ymax + 2.35,
            gap_text,
            color=COLORS["Gap"],
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS])
    ax.set_ylim(0, max(tiny_vals.max(), mobile_vals.max()) * 1.25 + 2.0)
    style_axes(ax, xlabel=False, ylabel="Best AP points")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=2)
    ax.set_title(
        "Metrics at the best-AP checkpoint favor TinyViM-B across localization and object scales",
        loc="left",
        pad=10.0,
        fontsize=10.2,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.155, top=0.84)
    save_all(fig, out_dir, "figure_2_visdrone_best_metric_summary")


def plot_best_latest_gap(best_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.65), gridspec_kw={"width_ratios": [1.65, 1.0]})

    metric_subset = [("AP", "AP"), ("AP50", "AP50"), ("AP75", "AP75")]
    ax = axes[0]
    x = np.arange(len(metric_subset))
    offsets = [-0.18, 0.18]
    for mi, method in enumerate(METHOD_ORDER):
        row = best_df[best_df["method"] == method].iloc[0]
        best_vals = np.array([row[f"best_{m}"] for m, _ in metric_subset], dtype=float)
        last_vals = np.array([row[f"latest_{m}"] for m, _ in metric_subset], dtype=float)
        color = COLORS[method]
        for i, (best_v, last_v) in enumerate(zip(best_vals, last_vals)):
            xpos = x[i] + offsets[mi]
            ax.plot([xpos, xpos], [last_v, best_v], color=color, lw=2.2, alpha=0.9)
            ax.scatter([xpos], [best_v], s=42, facecolor="white", edgecolor=color, linewidth=1.35, zorder=4)
            ax.scatter([xpos], [last_v], s=32, facecolor=color, edgecolor=color, linewidth=0.6, zorder=4)
            ax.text(xpos, best_v + 0.75, f"{best_v:.1f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_subset])
    ax.set_ylim(0, max(best_df[[f"best_{m}" for m, _ in metric_subset]].to_numpy().max() * 1.25, 1.0))
    style_axes(ax, ylabel="AP points")
    ax.set_title("Best point and final checkpoint", loc="left", pad=8.0, fontsize=9.5, fontweight="bold")
    panel_label(ax, "a")

    ax = axes[1]
    y = np.arange(len(METHOD_ORDER))
    latest_epochs = []
    best_epochs = []
    for method in METHOD_ORDER:
        row = best_df[best_df["method"] == method].iloc[0]
        latest_epochs.append(float(row["latest_epoch"]))
        best_epochs.append(float(row["best_AP_epoch"]))
    for i, method in enumerate(METHOD_ORDER):
        color = COLORS[method]
        ax.barh(i, latest_epochs[i], color="#EFEFEF", edgecolor=COLORS["Axis"], linewidth=0.45, height=0.46)
        ax.scatter([best_epochs[i]], [i], s=62, facecolor="white", edgecolor=color, linewidth=1.55, zorder=4)
        ax.text(latest_epochs[i] + 2.0, i, f"{int(latest_epochs[i])} val epochs", va="center", ha="left", fontsize=7.0)
        ax.text(best_epochs[i], i - 0.34, f"best AP @ {int(best_epochs[i])}", va="top", ha="center", fontsize=6.8, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(METHOD_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Validation epoch")
    ax.grid(axis="x", which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.75, pad=2.5)
    ax.set_title("Observed training span", loc="left", pad=8.0, fontsize=9.5, fontweight="bold")
    panel_label(ax, "b")

    custom = [
        plt.Line2D([0], [0], color=COLORS[METHOD_ORDER[0]], lw=2.2, marker="o", mfc="white", mec=COLORS[METHOD_ORDER[0]], label=METHOD_ORDER[0]),
        plt.Line2D([0], [0], color=COLORS[METHOD_ORDER[1]], lw=2.2, marker="s", mfc="white", mec=COLORS[METHOD_ORDER[1]], label=METHOD_ORDER[1]),
        plt.Line2D([0], [0], color=COLORS["Muted"], lw=0, marker="o", mfc="white", mec=COLORS["Muted"], label="Best checkpoint"),
        plt.Line2D([0], [0], color=COLORS["Muted"], lw=0, marker="o", mfc=COLORS["Muted"], mec=COLORS["Muted"], label="Final checkpoint"),
    ]
    fig.legend(custom, [h.get_label() for h in custom], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.012))
    fig.suptitle(
        "TinyViM-B still has the stronger checkpoint even after late-epoch decay",
        y=1.085,
        fontsize=10.4,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.80, wspace=0.32)
    save_all(fig, out_dir, "figure_3_visdrone_best_vs_final")


def write_readme(out_dir: Path, summary: pd.DataFrame, best_df: pd.DataFrame) -> None:
    tiny = best_df[best_df["method"] == METHOD_ORDER[0]].iloc[0]
    mobile = best_df[best_df["method"] == METHOD_ORDER[1]].iloc[0]
    lines = [
        "# VisDrone2019 dual experiment figures",
        "",
        "Data source:",
        f"- {CURVES_CSV}",
        f"- {SUMMARY_CSV}",
        "",
        "Key numbers:",
        f"- TinyViM-B + RetinaNet: best AP {tiny['best_AP']:.1f} at epoch {int(tiny['best_AP_epoch'])}; final AP {tiny['latest_AP']:.1f} at epoch {int(tiny['latest_epoch'])}.",
        f"- MobileMamba-B1 + RetinaNet: best AP {mobile['best_AP']:.1f} at epoch {int(mobile['best_AP_epoch'])}; final AP {mobile['latest_AP']:.1f} at epoch {int(mobile['latest_epoch'])}.",
        f"- Best AP gap: TinyViM-B is +{tiny['best_AP'] - mobile['best_AP']:.1f} AP points over MobileMamba-B1.",
        "",
        "Generated files:",
    ]
    for path in sorted(out_dir.glob("figure_*.*")):
        lines.append(f"- {path}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary.to_csv(out_dir / "source_dual_current_summary.csv", index=False)
    best_df.to_csv(out_dir / "derived_best_latest_summary.csv", index=False)


def main() -> None:
    setup_style()
    curves, summary = load_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = OUT_ROOT / f"visdrone_dual_cvpr_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_df = best_summary_from_curves(curves)
    plot_validation_curves(curves, out_dir)
    plot_best_metric_summary(summary, out_dir)
    plot_best_latest_gap(best_df, out_dir)
    write_readme(out_dir, summary, best_df)

    print(out_dir)
    for path in sorted(out_dir.glob("figure_*.*")):
        print(path)


if __name__ == "__main__":
    main()
