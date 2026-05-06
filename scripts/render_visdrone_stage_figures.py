#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from publication_style import (
    BACKGROUND,
    GRID,
    MODEL_COLORS,
    MUTED,
    PANEL,
    RANK_COLORS,
    TEXT,
    apply_publication_style,
    fmt_percent,
    metric_to_percent,
)


DISPLAY_NAMES = {
    "TinyViM_B": "TinyViM-B",
    "HybridMamba-Base_B": "HybridMamba-Base-B",
    "HybridMambaDet_B": "HybridMambaDet-B",
}

METRIC_COLUMNS = [
    ("bbox_mAP", "AP"),
    ("bbox_mAP_50", "AP50"),
    ("bbox_mAP_75", "AP75"),
    ("bbox_mAP_s", "AP-S"),
    ("bbox_mAP_m", "AP-M"),
    ("bbox_mAP_l", "AP-L"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render polished stage figures for the VisDrone main comparison.")
    parser.add_argument("--input-csv", type=Path, default=Path("artifacts/tables/visdrone_stage_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rank_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    lookup: dict[tuple[str, str], int] = {}
    for key, _ in METRIC_COLUMNS:
        ranked = sorted(rows, key=lambda row: float(row[key]), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            lookup[(row["model"], key)] = rank
    return lookup


def add_panel_tag(ax: plt.Axes, tag: str, title: str) -> None:
    ax.text(
        0.0,
        1.08,
        f"({tag})  {title}",
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=TEXT,
    )


def render_table(rows: list[dict[str, str]], output_dir: Path) -> None:
    rank_map = rank_lookup(rows)
    fig, ax = plt.subplots(figsize=(12.6, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = "Stage-I Quantitative Comparison on VisDrone2019-DET Validation"
    subtitle = "All runs use RetinaNet + FPN with the same 1x schedule. Ranked highlights follow 1st / 2nd / 3rd."
    ax.text(0.02, 0.935, title, fontsize=16.5, fontweight="bold", ha="left", va="center")
    ax.text(0.02, 0.865, subtitle, fontsize=11.0, color=MUTED, ha="left", va="center")

    left = 0.02
    right = 0.98
    top = 0.74
    row_h = 0.16
    method_w = 0.33
    metric_w = (right - left - method_w) / len(METRIC_COLUMNS)

    ax.plot([left, right], [top + 0.08, top + 0.08], color=TEXT, lw=1.2)
    ax.plot([left, right], [top - 0.02, top - 0.02], color=TEXT, lw=0.9)
    ax.plot([left, right], [top - row_h * len(rows) - 0.02, top - row_h * len(rows) - 0.02], color=TEXT, lw=1.2)

    ax.text(left + 0.008, top + 0.02, "Methods", fontsize=12.5, fontweight="bold", ha="left", va="center")
    for idx, (_, label) in enumerate(METRIC_COLUMNS):
        x0 = left + method_w + idx * metric_w
        ax.text(x0 + metric_w / 2, top + 0.02, label, fontsize=12.5, fontweight="bold", ha="center", va="center")

    for row_idx, row in enumerate(rows):
        y_center = top - row_h * row_idx - row_h / 2 - 0.02
        if row_idx < len(rows) - 1:
            ax.plot([left, right], [top - row_h * (row_idx + 1) - 0.02, top - row_h * (row_idx + 1) - 0.02], color=GRID, lw=0.8)

        method = DISPLAY_NAMES[row["model"]]
        ax.text(left + 0.008, y_center + 0.035, method, fontsize=13.5, fontweight="bold", ha="left", va="center")
        ax.text(left + 0.008, y_center - 0.032, row["variant"], fontsize=10.5, color=MUTED, ha="left", va="center")

        for idx, (metric_key, _) in enumerate(METRIC_COLUMNS):
            x0 = left + method_w + idx * metric_w + 0.01
            y0 = y_center - 0.047
            rank = rank_map[(row["model"], metric_key)]
            patch = FancyBboxPatch(
                (x0, y0),
                metric_w - 0.02,
                0.092,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                linewidth=0.0,
                facecolor=RANK_COLORS[rank],
                alpha=0.95,
            )
            ax.add_patch(patch)
            ax.text(
                x0 + (metric_w - 0.02) / 2,
                y_center,
                fmt_percent(float(row[metric_key])),
                fontsize=12.5,
                fontweight="bold" if rank == 1 else "normal",
                ha="center",
                va="center",
            )

    ax.text(
        left,
        0.085,
        "Current evidence is not final: the low-frequency-only variant is strongest, so the detail branch still needs tuning before paper claims are widened.",
        fontsize=10.2,
        color=MUTED,
        ha="left",
        va="center",
    )

    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"visdrone_stage_table_publication.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_grouped_metrics(rows: list[dict[str, str]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    add_panel_tag(ax, "a", "Absolute Validation Performance")
    metrics = [label for _, label in METRIC_COLUMNS]
    x = list(range(len(metrics)))
    width = 0.22

    for offset_idx, row in enumerate(rows):
        values = [metric_to_percent(float(row[key])) for key, _ in METRIC_COLUMNS]
        xs = [position + (offset_idx - 1) * width for position in x]
        bars = ax.bar(
            xs,
            values,
            width=width,
            color=MODEL_COLORS[row["model"]],
            edgecolor="white",
            linewidth=1.0,
            label=DISPLAY_NAMES[row["model"]],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.1f}", ha="center", va="bottom", fontsize=10.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11.5)
    ax.set_ylabel("AP (%)", fontsize=12.5)
    ax.set_ylim(0, 7.1)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.grid(axis="y")

    ax.text(
        0.64,
        0.74,
        "Stage note:\nHybridMambaDet improves the baseline,\n"
        "but it still trails the low-frequency-only variant.",
        transform=ax.transAxes,
        fontsize=10.2,
        color=MUTED,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL, edgecolor=GRID, linewidth=0.8),
    )

    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"visdrone_stage_metrics_publication.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_gain_plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    baseline = next(row for row in rows if row["model"] == "TinyViM_B")
    compare_rows = [row for row in rows if row["model"] != "TinyViM_B"]
    metrics = [("bbox_mAP", "AP"), ("bbox_mAP_50", "AP50"), ("bbox_mAP_75", "AP75"), ("bbox_mAP_s", "AP-S"), ("bbox_mAP_m", "AP-M"), ("bbox_mAP_l", "AP-L")]
    y_positions = list(range(len(metrics)))[::-1]

    fig, ax = plt.subplots(figsize=(9.3, 5.6))
    add_panel_tag(ax, "b", "Gain vs. TinyViM-B Baseline")
    ax.axvline(0.0, color="#5a5148", linestyle="--", linewidth=1.2)

    compare_offsets = {"HybridMamba-Base_B": 0.16, "HybridMambaDet_B": -0.16}
    for row in compare_rows:
        color = MODEL_COLORS[row["model"]]
        offset = compare_offsets[row["model"]]
        for idx, (metric_key, metric_label) in enumerate(metrics):
            y = y_positions[idx] + offset
            delta = metric_to_percent(float(row[metric_key]) - float(baseline[metric_key]))
            ax.plot([0, delta], [y, y], color=color, lw=2.0, alpha=0.9)
            ax.scatter(delta, y, s=64, color=color, zorder=4)
            ax.text(delta + 0.06, y, f"{delta:+.1f}", fontsize=10.5, va="center", ha="left", color=color)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in metrics], fontsize=11.5)
    ax.set_xlabel("Delta vs. TinyViM-B (pts)", fontsize=12.5)
    ax.set_xlim(-0.2, 2.4)
    ax.set_ylim(-0.6, len(metrics) - 0.4)
    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[name], lw=2.4, marker="o", markersize=6) for name in ("HybridMamba-Base_B", "HybridMambaDet_B")]
    ax.legend(handles, [DISPLAY_NAMES["HybridMamba-Base_B"], DISPLAY_NAMES["HybridMambaDet_B"]], loc="lower right")
    ax.grid(axis="x")

    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"visdrone_stage_gain_publication.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_summary_json(rows: list[dict[str, str]], output_dir: Path) -> None:
    payload = {
        "best_model_currently": max(rows, key=lambda row: float(row["bbox_mAP"]))["model"],
        "results": rows,
    }
    (output_dir / "visdrone_stage_figure_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = (repo_root / args.input_csv).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_publication_style()
    rows = load_rows(input_csv)

    render_table(rows, output_dir)
    render_grouped_metrics(rows, output_dir)
    render_gain_plot(rows, output_dir)
    render_summary_json(rows, output_dir)

    print(
        json.dumps(
            {
                "table": str(output_dir / "visdrone_stage_table_publication.png"),
                "metrics": str(output_dir / "visdrone_stage_metrics_publication.png"),
                "gain": str(output_dir / "visdrone_stage_gain_publication.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
