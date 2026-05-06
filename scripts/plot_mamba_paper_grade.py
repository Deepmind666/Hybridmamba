#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "artifacts" / "analysis" / "mamba_current_20260430_0105"
SUMMARY_CSV = ANALYSIS_DIR / "current_results_summary.csv"
CURVES_CSV = ANALYSIS_DIR / "validation_curves.csv"

OUT_STEM = "figure_mamba_paper_grade_main"
TABLE_CSV = ANALYSIS_DIR / "paper_grade_evidence_table.csv"
TABLE_MD = ANALYSIS_DIR / "paper_grade_evidence_table.md"

COLORS = {
    "TinyViM-B": "#3B4992",
    "HybridMamba-Base": "#8C8C8C",
    "HybridMambaDet": "#008B45",
    "Fusion alpha=1.0": "#E64B35",
    "Fusion alpha=0.5": "#E69F00",
    "Stage shallow": "#7E6148",
}

FAT_ORDER = ["TinyViM-B", "HybridMambaDet", "Fusion alpha=1.0"]
LOCAL_ORDER = ["Fusion alpha=0.5", "Stage shallow"]
LABELS = {
    "TinyViM-B": "TinyViM",
    "HybridMamba-Base": "Base",
    "HybridMambaDet": "Det",
    "Fusion alpha=1.0": "F10",
    "Fusion alpha=0.5": "F05",
    "Stage shallow": "Stage",
    "TinyViM-B local best": "TinyViM-L",
}

METRICS = [
    ("AP", "best_AP"),
    ("AP75", "best_AP75"),
    ("AP_S", "best_AP_S"),
    ("AP_M", "best_AP_M"),
    ("AP_L", "best_AP_L"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_by_label(rows: list[dict[str, str]], label: str) -> dict[str, str]:
    return next(row for row in rows if row["label"] == label)


def val(row: dict[str, str], key: str) -> float:
    return float(row[key]) * 100.0


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "font.size": 6.8,
            "axes.titlesize": 7.1,
            "axes.labelsize": 6.9,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=9.2, weight="bold", va="top", ha="left")


def plot_fat_delta(ax: plt.Axes, summary: list[dict[str, str]]) -> None:
    base = row_by_label(summary, "HybridMamba-Base")
    y_base = list(range(len(METRICS)))[::-1]
    offsets = {"TinyViM-B": 0.22, "HybridMambaDet": 0.0, "Fusion alpha=1.0": -0.22}

    ax.axvline(0, color="#222222", linewidth=0.8)
    for x in [-2, -1, 1]:
        ax.axvline(x, color="#E3E3E3", linewidth=0.55, zorder=0)

    for label in FAT_ORDER:
        row = row_by_label(summary, label)
        xs = [val(row, key) - val(base, key) for _, key in METRICS]
        ys = [y + offsets[label] for y in y_base]
        linestyle = "-" if label != "Fusion alpha=1.0" else (0, (3, 2))
        for x, y in zip(xs, ys):
            ax.plot([0, x], [y, y], color=COLORS[label], linewidth=1.0, linestyle=linestyle, alpha=0.78)
        ax.scatter(xs, ys, s=22, color=COLORS[label], edgecolor="white", linewidth=0.45, zorder=3, label=LABELS[label])

    ax.set_yticks(y_base)
    ax.set_yticklabels([name for name, _ in METRICS])
    ax.set_xlim(-2.25, 0.85)
    ax.set_xlabel("Delta from Base (AP points)")
    ax.set_title("Same-machine effect size", loc="left", pad=4)
    ax.legend(frameon=False, loc="upper left", ncols=1, handletextpad=0.35, borderpad=0.1, labelspacing=0.25)
    panel_letter(ax, "a")


def plot_fat_trajectory(ax: plt.Axes, summary: list[dict[str, str]], curves: list[dict[str, str]]) -> None:
    show_labels = ["TinyViM-B", "HybridMamba-Base", "HybridMambaDet", "Fusion alpha=1.0"]
    label_offsets = {"TinyViM-B": 0.06, "HybridMamba-Base": -0.08, "HybridMambaDet": 0.05, "Fusion alpha=1.0": 0.06}

    for label in show_labels:
        pts = [row for row in curves if row["label"] == label and int(row["epoch"]) >= 8]
        pts.sort(key=lambda row: int(row["epoch"]))
        xs = [int(row["epoch"]) for row in pts]
        ys = [float(row["AP"]) * 100.0 for row in pts]
        linestyle = "-" if label != "Fusion alpha=1.0" else (0, (3, 2))
        ax.plot(xs, ys, color=COLORS[label], linewidth=1.15, linestyle=linestyle, alpha=0.96)
        ax.scatter(xs, ys, s=9, color=COLORS[label], edgecolor="white", linewidth=0.28, zorder=3)

        best = row_by_label(summary, label)
        best_epoch = int(float(best["best_epoch"]))
        best_ap = val(best, "best_AP")
        if best_epoch >= 8:
            ax.scatter([best_epoch], [best_ap], s=38, facecolor="white", edgecolor=COLORS[label], linewidth=1.05, zorder=4)
        ax.text(xs[-1] + 0.45, ys[-1] + label_offsets.get(label, 0.0), LABELS[label], color=COLORS[label], fontsize=6.6, va="center")

    ax.set_xlim(8, 33.8)
    ax.set_ylim(18.85, 20.55)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AP (points)")
    ax.set_title("Fat validation trajectory after warm-up", loc="left", pad=4)
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.55)
    ax.text(0.99, 0.05, "open circles mark best checkpoints", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color="#666666")
    panel_letter(ax, "b")


def plot_local_context(ax: plt.Axes, summary: list[dict[str, str]]) -> None:
    local_ref = row_by_label(summary, "TinyViM-B local best")
    metrics = [("AP", "best_AP"), ("AP_S", "best_AP_S"), ("AP_L", "best_AP_L")]
    y_base = list(range(len(metrics)))[::-1]
    offsets = {"Fusion alpha=0.5": 0.13, "Stage shallow": -0.13}

    ax.axvline(0, color="#222222", linewidth=0.8)
    for x in [-1, 1]:
        ax.axvline(x, color="#E5E5E5", linewidth=0.5, zorder=0)

    for label in LOCAL_ORDER:
        row = row_by_label(summary, label)
        xs = [val(row, key) - val(local_ref, key) for _, key in metrics]
        ys = [y + offsets[label] for y in y_base]
        for x, y in zip(xs, ys):
            ax.plot([0, x], [y, y], color=COLORS[label], linewidth=0.9, alpha=0.72)
        ax.scatter(xs, ys, s=20, color=COLORS[label], edgecolor="white", linewidth=0.42, zorder=3, label=LABELS[label])
        for x, y in zip(xs, ys):
            ax.text(x + (0.06 if x >= 0 else -0.06), y, f"{x:+.1f}", ha="left" if x >= 0 else "right", va="center", fontsize=6.0)

    ax.set_yticks(y_base)
    ax.set_yticklabels([name for name, _ in metrics])
    ax.set_xlim(-1.05, 1.45)
    ax.set_xlabel("Delta from local TinyViM (AP points)")
    ax.set_title("Local-only ablation context", loc="left", pad=4)
    ax.legend(frameon=False, loc="upper right", ncols=1, handletextpad=0.25, borderpad=0.1, labelspacing=0.25)
    panel_letter(ax, "c")


def plot_fat_absolute_table(ax: plt.Axes, summary: list[dict[str, str]]) -> None:
    labels = ["TinyViM-B", "HybridMamba-Base", "HybridMambaDet", "Fusion alpha=1.0"]
    columns = [("AP", "best_AP"), ("AP_S", "best_AP_S"), ("AP_L", "best_AP_L")]
    ax.axis("off")
    ax.set_title("Absolute best metrics on Fat", loc="left", pad=4)

    x_model = 0.02
    x_cols = [0.50, 0.70, 0.90]
    y_head = 0.70
    row_gap = 0.16
    ax.text(x_model, y_head, "Model", fontsize=6.3, weight="bold", ha="left", va="center")
    for x, (name, _) in zip(x_cols, columns):
        ax.text(x, y_head, name, fontsize=6.3, weight="bold", ha="center", va="center")
    ax.plot([0.02, 0.96], [0.75, 0.75], color="#111111", linewidth=0.65)

    for idx, label in enumerate(labels):
        y = y_head - (idx + 1) * row_gap
        row = row_by_label(summary, label)
        ax.text(x_model, y, LABELS[label], fontsize=6.25, color=COLORS[label], ha="left", va="center")
        for x, (_, key) in zip(x_cols, columns):
            ax.text(x, y, f"{val(row, key):.1f}", fontsize=6.25, ha="center", va="center")
        if idx in {1, 3}:
            ax.plot([0.02, 0.96], [y - row_gap / 2, y - row_gap / 2], color="#E8E8E8", linewidth=0.45)

    panel_letter(ax, "d")


def write_evidence_table(summary: list[dict[str, str]]) -> None:
    rows: list[dict[str, Any]] = []
    verdict = {
        "TinyViM-B": "Fat baseline",
        "HybridMamba-Base": "Mamba baseline",
        "HybridMambaDet": "weak positive AP_S, AP_L penalty",
        "Fusion alpha=1.0": "no clean Fat win",
        "Fusion alpha=0.5": "local-only signal",
        "Stage shallow": "negative/weak local branch",
        "TinyViM-B local best": "local reference only",
    }
    for row in summary:
        rows.append(
            {
                "run": row["label"],
                "host": row["host"],
                "fat_comparable": row["host"] == "Fat",
                "status": row["status"],
                "best_epoch": row["best_epoch"],
                "best_AP_points": f"{val(row, 'best_AP'):.1f}",
                "best_AP_S_points": f"{val(row, 'best_AP_S'):.1f}",
                "best_AP_L_points": f"{val(row, 'best_AP_L'):.1f}",
                "verdict": verdict.get(row["label"], ""),
            }
        )
    with TABLE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| Run | Host | Fat comparable | Status | Best epoch | AP | AP_S | AP_L | Verdict |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(
            "| {run} | {host} | {fat_comparable} | {status} | {best_epoch} | {best_AP_points} | {best_AP_S_points} | {best_AP_L_points} | {verdict} |".format(
                **row
            )
        )
    TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(ANALYSIS_DIR / f"{OUT_STEM}.{ext}", bbox_inches="tight", dpi=300)


def main() -> None:
    setup_style()
    summary = read_csv(SUMMARY_CSV)
    curves = read_csv(CURVES_CSV)

    fig = plt.figure(figsize=(7.25, 4.85))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.08),
        height_ratios=(1.0, 0.88),
        left=0.075,
        right=0.985,
        top=0.855,
        bottom=0.125,
        wspace=0.34,
        hspace=0.58,
    )

    fig.text(0.075, 0.962, "HybridMamba evidence is currently weak on VisDrone", ha="left", va="top", fontsize=7.8, weight="bold")
    fig.text(
        0.075,
        0.935,
        "Fat-machine comparisons show only a small AP_S gain and a large AP_L penalty; local ablations are shown separately.",
        ha="left",
        va="top",
        fontsize=5.7,
        color="#666666",
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 0])

    plot_fat_delta(ax_a, summary)
    plot_fat_trajectory(ax_b, summary, curves)
    plot_local_context(ax_c, summary)
    plot_fat_absolute_table(ax_d, summary)

    save_figure(fig)
    write_evidence_table(summary)
    plt.close(fig)
    print(f"Wrote {ANALYSIS_DIR / (OUT_STEM + '.png')}")
    print(f"Wrote {TABLE_CSV}")


if __name__ == "__main__":
    main()
