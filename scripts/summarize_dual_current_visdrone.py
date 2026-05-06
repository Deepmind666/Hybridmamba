#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D


METRICS = [
    ("AP", "coco/bbox_mAP"),
    ("AP50", "coco/bbox_mAP_50"),
    ("AP75", "coco/bbox_mAP_75"),
    ("AP_S", "coco/bbox_mAP_s"),
    ("AP_M", "coco/bbox_mAP_m"),
    ("AP_L", "coco/bbox_mAP_l"),
]

COLORS = {
    "TinyViM-B + RetinaNet": "#2D83BD",
    "MobileMamba-B1 + RetinaNet": "#36A657",
    "Best checkpoint": "#D15B9A",
    "Latest checkpoint": "#5A5A5A",
}

VAL_RE = re.compile(r"Epoch\(val\)\s+\[(?P<epoch>\d+)\].*?(?P<metrics>coco/bbox_mAP:.*)")
METRIC_RE = re.compile(r"(coco/bbox_mAP(?:_50|_75|_s|_m|_l)?):\s*(-?\d+(?:\.\d+)?)")


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": ["DejaVu Sans", "Arial"],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.72,
            "axes.edgecolor": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.72,
            "ytick.major.width": 0.72,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.48,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_log(path: Path, method: str) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = VAL_RE.search(line)
            if not match:
                continue
            metric_pairs = dict(METRIC_RE.findall(match.group("metrics")))
            if "coco/bbox_mAP" not in metric_pairs:
                continue
            row: dict[str, float | int | str] = {
                "method": method,
                "epoch": int(match.group("epoch")),
            }
            for short, key in METRICS:
                row[short] = float(metric_pairs.get(key, "nan")) * 100.0
            rows.append(row)
    rows.sort(key=lambda item: int(item["epoch"]))
    dedup: dict[int, dict[str, float | int | str]] = {}
    for row in rows:
        dedup[int(row["epoch"])] = row
    return [dedup[k] for k in sorted(dedup)]


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "epoch"] + [short for short, _ in METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows_by_method: dict[str, list[dict[str, float | int | str]]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for method, rows in rows_by_method.items():
        if not rows:
            continue
        best = max(rows, key=lambda item: float(item["AP"]))
        latest = rows[-1]
        summary.append(
            {
                "method": method,
                "n_val_epochs": str(len(rows)),
                "first_epoch": str(rows[0]["epoch"]),
                "latest_epoch": str(latest["epoch"]),
                "latest_AP": f"{float(latest['AP']):.3f}",
                "latest_AP50": f"{float(latest['AP50']):.3f}",
                "latest_AP75": f"{float(latest['AP75']):.3f}",
                "best_epoch": str(best["epoch"]),
                "best_AP": f"{float(best['AP']):.3f}",
                "best_AP50": f"{float(best['AP50']):.3f}",
                "best_AP75": f"{float(best['AP75']):.3f}",
                "best_AP_S": f"{float(best['AP_S']):.3f}",
                "best_AP_M": f"{float(best['AP_M']):.3f}",
                "best_AP_L": f"{float(best['AP_L']):.3f}",
            }
        )
    return summary


def write_summary(out_dir: Path, summary: list[dict[str, str]]) -> None:
    csv_path = out_dir / "dual_current_summary.csv"
    fields = list(summary[0].keys()) if summary else ["method"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "| Method | Val epochs | Latest epoch | Latest AP | Best epoch | Best AP | Best AP50 | Best AP75 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['n_val_epochs']} | {row['latest_epoch']} | {row['latest_AP']} | "
            f"{row['best_epoch']} | {row['best_AP']} | {row['best_AP50']} | {row['best_AP75']} |"
        )
    (out_dir / "dual_current_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_all(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.035)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.07, label, transform=ax.transAxes, ha="left", va="top", fontsize=9.6, weight="bold")


def plot_trajectories(out_dir: Path, rows_by_method: dict[str, list[dict[str, float | int | str]]]) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.65), sharex=False)
    plot_specs = [("AP", "AP"), ("AP50", "AP50"), ("AP75", "AP75")]

    for ax, (metric, title), letter in zip(axes, plot_specs, ["a", "b", "c"]):
        ymin = 1e9
        ymax = -1e9
        for method, rows in rows_by_method.items():
            xs = [int(row["epoch"]) for row in rows]
            ys = [float(row[metric]) for row in rows]
            if not xs:
                continue
            color = COLORS[method]
            ymin = min(ymin, min(ys))
            ymax = max(ymax, max(ys))
            ax.plot(xs, ys, color=color, linewidth=1.65, solid_capstyle="round", label=method)
            ax.scatter(xs, ys, s=9, color=color, edgecolor="white", linewidth=0.25, zorder=3)
            best = max(rows, key=lambda item: float(item["AP"]))
            if metric == "AP":
                ax.scatter(
                    [int(best["epoch"])],
                    [float(best[metric])],
                    s=42,
                    facecolor="white",
                    edgecolor=COLORS["Best checkpoint"],
                    linewidth=1.1,
                    zorder=5,
                )
        margin = max(0.4, (ymax - ymin) * 0.14)
        ax.set_ylim(max(0, ymin - margin), ymax + margin)
        ax.set_title(title, loc="left", pad=4)
        ax.set_xlabel("Validation epoch")
        ax.grid(axis="y")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        panel_label(ax, letter)
    axes[0].set_ylabel("COCO AP points")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), handlelength=2.2)
    fig.suptitle("Current VisDrone validation trajectories", x=0.07, y=1.04, ha="left", fontsize=9.0, weight="bold")
    save_all(fig, out_dir, "figure_1_validation_trajectories")
    plt.close(fig)


def plot_best_latest(out_dir: Path, rows_by_method: dict[str, list[dict[str, float | int | str]]]) -> None:
    apply_style()
    metrics = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), sharey=True)

    for ax, (method, rows), letter in zip(axes, rows_by_method.items(), ["a", "b"]):
        best = max(rows, key=lambda item: float(item["AP"]))
        latest = rows[-1]
        y = list(range(len(metrics)))[::-1]
        best_values = [float(best[m]) for m in metrics]
        latest_values = [float(latest[m]) for m in metrics]
        for yy, lo, hi in zip(y, latest_values, best_values):
            ax.plot([lo, hi], [yy, yy], color="#CFCFCF", linewidth=2.1, solid_capstyle="round", zorder=1)
        ax.scatter(latest_values, y, color=COLORS["Latest checkpoint"], s=28, zorder=3)
        ax.scatter(best_values, y, facecolor="white", edgecolor=COLORS["Best checkpoint"], linewidth=1.25, s=42, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(metrics)
        ax.set_xlabel("AP points")
        ax.set_title(
            f"{method}\nlatest epoch {latest['epoch']}; best epoch {best['epoch']}",
            loc="left",
            pad=5,
            linespacing=1.12,
        )
        ax.grid(axis="x")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        panel_label(ax, letter)
    axes[0].set_ylabel("Metric")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["Latest checkpoint"], markeredgecolor=COLORS["Latest checkpoint"], markersize=5.5, label="Latest checkpoint"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["Best checkpoint"], markeredgewidth=1.1, markersize=6.4, label="Best checkpoint"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), handletextpad=0.5, columnspacing=1.5)
    fig.suptitle("Best checkpoint versus latest validation", x=0.07, y=1.03, ha="left", fontsize=9.0, weight="bold")
    fig.subplots_adjust(bottom=0.19, top=0.80, wspace=0.24)
    save_all(fig, out_dir, "figure_2_best_latest_metrics")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-log", type=Path, required=True)
    parser.add_argument("--fat-log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_method = {
        "TinyViM-B + RetinaNet": parse_log(args.local_log, "TinyViM-B + RetinaNet"),
        "MobileMamba-B1 + RetinaNet": parse_log(args.fat_log, "MobileMamba-B1 + RetinaNet"),
    }
    all_rows = [row for rows in rows_by_method.values() for row in rows]
    write_csv(out_dir / "dual_current_validation_curves.csv", all_rows)
    summary = summarize(rows_by_method)
    write_summary(out_dir, summary)
    plot_trajectories(out_dir, rows_by_method)
    plot_best_latest(out_dir, rows_by_method)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
