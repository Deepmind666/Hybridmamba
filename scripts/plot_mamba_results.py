#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt

from publication_style import MUTED, TEXT, apply_publication_style


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "artifacts" / "analysis" / "mamba_current_20260430_0105"
REMOTE_CACHE = ANALYSIS_DIR / "remote_cache"
RUNS_ROOT = ROOT / "artifacts" / "runs"

METRICS = [
    ("AP", "coco/bbox_mAP"),
    ("AP50", "coco/bbox_mAP_50"),
    ("AP75", "coco/bbox_mAP_75"),
    ("AP_S", "coco/bbox_mAP_s"),
    ("AP_M", "coco/bbox_mAP_m"),
    ("AP_L", "coco/bbox_mAP_l"),
]


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_id: str
    host: str
    run_dir: Path
    status: str
    include_main: bool = False
    include_curve: bool = True
    note: str = ""


RUNS = [
    RunSpec(
        "TinyViM-B",
        "fat_tinyvim1x_stable_20260428_124251",
        "Fat",
        REMOTE_CACHE / "fat_tinyvim1x_stable_20260428_124251",
        "complete",
        include_main=True,
    ),
    RunSpec(
        "HybridMamba-Base",
        "fat_hybridmamba_base1x_stable_copyfix_20260429_1323",
        "Fat",
        REMOTE_CACHE / "fat_hybridmamba_base1x_stable_copyfix_20260429_1323",
        "complete",
        include_main=True,
    ),
    RunSpec(
        "HybridMambaDet",
        "fat_hybridmambadet1x_stable_20260425_1050",
        "Fat",
        REMOTE_CACHE / "fat_hybridmambadet1x_stable_20260425_1050",
        "complete",
        include_main=True,
    ),
    RunSpec(
        "Fusion alpha=1.0",
        "fat_hybridmambadet_fusion10_stable_retry_20260429_2045",
        "Fat",
        REMOTE_CACHE / "fat_hybridmambadet_fusion10_stable_retry_20260429_2045",
        "stopped",
        include_main=True,
        note="stopped on request; best epoch 12 and latest epoch 25 are both below the main Fat baseline",
    ),
    RunSpec(
        "Stage shallow",
        "local_hybridmambadet_stage01_stable_20260429_052416_a1",
        "Local",
        RUNS_ROOT / "local_hybridmambadet_stage01_stable_20260429_052416_a1",
        "stopped",
        include_main=False,
        note="stopped on request; resumed evidence is clearly negative",
    ),
    RunSpec(
        "TinyViM-B local best",
        "local_tinyvim1x_stable_adaptive_mem22_20260428_143121",
        "Local",
        RUNS_ROOT / "local_tinyvim1x_stable_adaptive_mem22_20260428_143121",
        "partial",
        include_main=False,
        note="best absolute local checkpoint, not used as main Fat-comparable evidence",
    ),
    RunSpec(
        "Fusion alpha=0.5",
        "local_hybridmambadet_fusion05_stable_20260430_0100_a1",
        "Local",
        RUNS_ROOT / "local_hybridmambadet_fusion05_stable_20260430_0100_a1",
        "complete",
        include_main=False,
        note="early stopped at epoch 33; strongest local-only Mamba ablation point so far",
    ),
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_json_loose(path: Path) -> dict[str, Any] | None:
    text = read_text(path)
    if not text.strip():
        return None
    if "#< CLIXML" in text:
        text = text.split("#< CLIXML", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


VAL_RE = re.compile(
    r"Epoch\(val\)\s+\[\s*(?P<epoch>\d+)\]\[\s*\d+/\d+\]\s+"
    r"coco/bbox_mAP:\s+(?P<ap>[0-9.]+)\s+"
    r"coco/bbox_mAP_50:\s+(?P<ap50>[0-9.]+)\s+"
    r"coco/bbox_mAP_75:\s+(?P<ap75>[0-9.]+)\s+"
    r"coco/bbox_mAP_s:\s+(?P<aps>[0-9.]+)\s+"
    r"coco/bbox_mAP_m:\s+(?P<apm>[0-9.]+)\s+"
    r"coco/bbox_mAP_l:\s+(?P<apl>[0-9.]+)"
)


def parse_curve(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in VAL_RE.finditer(read_text(path)):
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "AP": float(match.group("ap")),
                "AP50": float(match.group("ap50")),
                "AP75": float(match.group("ap75")),
                "AP_S": float(match.group("aps")),
                "AP_M": float(match.group("apm")),
                "AP_L": float(match.group("apl")),
            }
        )
    return rows


def metric_value(metrics: dict[str, Any] | None, section: str, key: str) -> float | None:
    if not metrics:
        return None
    block = metrics.get(section) or {}
    value = block.get(key)
    return float(value) if value is not None else None


def build_summary() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for spec in RUNS:
        metrics = load_json_loose(spec.run_dir / "eval_metrics.json")
        row: dict[str, Any] = {
            "label": spec.label,
            "run_id": spec.run_id,
            "host": spec.host,
            "status": spec.status,
            "include_main": spec.include_main,
            "best_epoch": metrics.get("best_epoch") if metrics else "",
            "latest_epoch": metrics.get("epoch") if metrics else "",
            "note": spec.note,
        }
        for short, key in METRICS:
            row[f"best_{short}"] = metric_value(metrics, "best", key)
            row[f"latest_{short}"] = metric_value(metrics, "latest", key)
        summary.append(row)

        if spec.include_curve:
            for point in parse_curve(spec.run_dir / "train.log"):
                curves.append(
                    {
                        "label": spec.label,
                        "run_id": spec.run_id,
                        "host": spec.host,
                        "status": spec.status,
                        **point,
                    }
                )
    return summary, curves


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def style() -> None:
    apply_publication_style()
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 450,
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "axes.titleweight": "normal",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


PALETTE = {
    "TinyViM-B": "#3B4992",
    "HybridMamba-Base": "#808180",
    "HybridMambaDet": "#008B45",
    "Fusion alpha=1.0": "#EE0000",
    "Stage shallow": "#631879",
    "TinyViM-B local best": "#008280",
    "Fusion alpha=0.5": "#E69F00",
}

METRIC_COLORS = {
    "AP": "#2F6B9A",
    "AP_S": "#2A9D8F",
    "AP_L": "#D97B2E",
}

STATUS_COLORS = {
    "complete": "#E8F3EE",
    "stopped": "#FBE9E7",
    "partial": "#EEF2FB",
}


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _row(summary: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return next(row for row in summary if row["label"] == label)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(ANALYSIS_DIR / f"{stem}.{ext}", bbox_inches="tight")


def plot_current(summary: list[dict[str, Any]], curves: list[dict[str, Any]]) -> None:
    style()

    short = {
        "TinyViM-B": "TinyViM",
        "HybridMamba-Base": "Base",
        "HybridMambaDet": "Det",
        "Fusion alpha=1.0": "F10",
        "Stage shallow": "Stage",
        "TinyViM-B local best": "TinyViM-L",
        "Fusion alpha=0.5": "F05",
    }

    fat_labels = ["TinyViM-B", "HybridMamba-Base", "HybridMambaDet", "Fusion alpha=1.0"]
    metric_panels = [
        ("AP", "best_AP", (19.4, 20.7)),
        ("AP_S", "best_AP_S", (11.8, 12.7)),
        ("AP_L", "best_AP_L", (31.8, 35.2)),
    ]

    fig = plt.figure(figsize=(12.2, 7.4), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 1.05),
        height_ratios=(1.0, 1.06),
        left=0.07,
        right=0.985,
        top=0.82,
        bottom=0.105,
        wspace=0.33,
        hspace=0.42,
    )

    fig.text(0.07, 0.965, "HybridMamba evidence snapshot on VisDrone", ha="left", va="top", fontsize=11.0, weight="bold")
    fig.text(
        0.07,
        0.925,
        "Best checkpoints; Fat-machine results are separated from local-only context. Values are COCO AP points.",
        ha="left",
        va="top",
        fontsize=7.8,
        color=MUTED,
    )

    # Panel a: absolute Fat-comparable metrics as small multiples.
    y_positions = list(range(len(fat_labels)))[::-1]
    for idx, (metric, key, xlim) in enumerate(metric_panels):
        ax = fig.add_subplot(gs[0, idx])
        values = [float(_row(summary, label)[key]) * 100.0 for label in fat_labels]
        base_value = float(_row(summary, "HybridMamba-Base")[key]) * 100.0
        ax.axvline(base_value, color="#6f6f6f", linewidth=0.85, linestyle=(0, (2, 2)), zorder=1)
        for y, label, value in zip(y_positions, fat_labels, values):
            if label != "HybridMamba-Base":
                ax.plot(
                    [base_value, value],
                    [y, y],
                    color=PALETTE[label],
                    linewidth=1.15,
                    alpha=0.65,
                    zorder=2,
                )
            ax.scatter(
                [value],
                [y],
                s=34,
                color=PALETTE[label],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
                alpha=0.98 if label != "Fusion alpha=1.0" else 0.86,
            )
            ha = "left" if value >= base_value else "right"
            offset = 0.025 if ha == "left" else -0.025
            ax.text(value + offset, y, f"{value:.1f}", va="center", ha=ha, fontsize=7.0)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, len(fat_labels) - 0.3)
        ax.set_title(metric, loc="left", pad=6)
        ax.grid(axis="x", color="#d8d8d8", linewidth=0.45)
        ax.tick_params(axis="y", length=0)
        if idx == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([short[label] for label in fat_labels])
            ax.text(-0.18, 1.06, "a", transform=ax.transAxes, fontsize=10.5, weight="bold", va="top")
        else:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([])
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("AP points")

    # Panel b: Fat-machine AP trajectories with direct line labels.
    ax_curve = fig.add_subplot(gs[1, 0:2])
    max_epoch = 0
    label_offsets = {
        "TinyViM-B": -0.0004,
        "HybridMamba-Base": -0.0012,
        "HybridMambaDet": 0.0009,
        "Fusion alpha=1.0": 0.0003,
    }
    for label in fat_labels:
        pts = [row for row in curves if row["label"] == label]
        if not pts:
            continue
        pts.sort(key=lambda row: row["epoch"])
        xs = [int(row["epoch"]) for row in pts]
        ys = [float(row["AP"]) * 100.0 for row in pts]
        max_epoch = max(max_epoch, max(xs))
        linestyle = "-" if _row(summary, label)["status"] == "complete" else (0, (3, 2))
        ax_curve.plot(xs, ys, linestyle=linestyle, linewidth=1.65, color=PALETTE[label], alpha=0.95)
        ax_curve.scatter(xs, ys, s=10, color=PALETTE[label], edgecolor="white", linewidth=0.35, zorder=3)
        ax_curve.text(
            xs[-1] + 0.45,
            ys[-1] + label_offsets.get(label, 0.0) * 100.0,
            short[label],
            color=PALETTE[label],
            fontsize=7.3,
            va="center",
        )
    ax_curve.text(-0.075, 1.08, "b", transform=ax_curve.transAxes, fontsize=10.5, weight="bold", va="top")
    ax_curve.set_title("Validation trajectories on the Fat machine", loc="left", pad=8)
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Validation AP points")
    ax_curve.set_xlim(0, max_epoch + 5)
    ax_curve.set_ylim(18.9, 20.9)
    ax_curve.grid(axis="y", color="#d8d8d8", linewidth=0.45)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)

    # Panel c: compact delta heatmap relative to HybridMamba-Base.
    ax_heat = fig.add_subplot(gs[1, 2])
    base = _row(summary, "HybridMamba-Base")
    delta_sources = ["HybridMambaDet", "Fusion alpha=1.0", "Fusion alpha=0.5", "Stage shallow"]
    delta_rows = ["Det", "F10", "F05 local", "Stage local"]
    delta_metrics = [
        ("AP", "best_AP"),
        ("AP75", "best_AP75"),
        ("AP_S", "best_AP_S"),
        ("AP_M", "best_AP_M"),
        ("AP_L", "best_AP_L"),
    ]
    heat = []
    for src in delta_sources:
        row = _row(summary, src)
        heat.append([(float(row[key]) - float(base[key])) * 100.0 for _, key in delta_metrics])

    cmap = LinearSegmentedColormap.from_list("journal_delta", ["#2166AC", "#F7F7F7", "#B2182B"])
    norm = TwoSlopeNorm(vmin=-2.2, vcenter=0.0, vmax=0.8)
    im = ax_heat.imshow(heat, cmap=cmap, norm=norm, aspect="auto")
    ax_heat.set_xticks(range(len(delta_metrics)))
    ax_heat.set_xticklabels([name for name, _ in delta_metrics], rotation=0)
    ax_heat.set_yticks(range(len(delta_rows)))
    ax_heat.set_yticklabels(delta_rows)
    ax_heat.set_title("Delta vs Base", loc="left", pad=8)
    ax_heat.text(-0.18, 1.08, "c", transform=ax_heat.transAxes, fontsize=10.5, weight="bold", va="top")
    ax_heat.set_xticks([x - 0.5 for x in range(1, len(delta_metrics))], minor=True)
    ax_heat.set_yticks([y - 0.5 for y in range(1, len(delta_rows))], minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.1)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    for y, row in enumerate(heat):
        for x, value in enumerate(row):
            color = "white" if abs(value) > 1.2 else TEXT
            ax_heat.text(x, y, f"{value:+.1f}", ha="center", va="center", fontsize=7.0, color=color)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.set_label("AP-point delta", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.8, length=2.5)

    fig.text(
        0.07,
        0.035,
        "Reading: Det matches TinyViM AP on Fat but only gives a small AP_S gain over Base; F05 is the strongest local-only point and still needs same-machine confirmation.",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
    )

    save_figure(fig, "figure_current_mamba_results")
    save_figure(fig, "figure_mamba_publication_summary")
    save_figure(fig, "figure_mamba_topjournal_summary")
    plt.close(fig)


def write_notes(summary: list[dict[str, Any]]) -> None:
    by_label = {row["label"]: row for row in summary}
    base = by_label["HybridMamba-Base"]
    tiny = by_label["TinyViM-B"]
    det = by_label["HybridMambaDet"]
    fusion10 = by_label["Fusion alpha=1.0"]
    stage = by_label["Stage shallow"]
    fusion05 = by_label["Fusion alpha=0.5"]

    def delta(row: dict[str, Any], key: str, ref: dict[str, Any] = base) -> float:
        return float(row[key]) - float(ref[key])

    lines = [
        "# Current Mamba Experiment Snapshot",
        "",
        "Main Fat-machine comparable results use best validation checkpoints.",
        "",
        "| Run | Status | Best epoch | Best AP | Latest AP | Best AP_S | Best AP_L |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [tiny, base, det, fusion10, fusion05, stage]:
        lines.append(
            "| {label} | {status} | {best_epoch} | {ap} | {latest} | {aps} | {apl} |".format(
                label=row["label"],
                status=row["status"],
                best_epoch=row["best_epoch"],
                ap=_fmt(row["best_AP"]),
                latest=_fmt(row["latest_AP"]),
                aps=_fmt(row["best_AP_S"]),
                apl=_fmt(row["best_AP_L"]),
            )
        )
    lines.extend(
        [
            "",
            "Key deltas against HybridMamba-Base (best checkpoints):",
            f"- HybridMambaDet: AP {delta(det, 'best_AP'):+.3f}, AP_S {delta(det, 'best_AP_S'):+.3f}, AP_L {delta(det, 'best_AP_L'):+.3f}.",
            f"- Fusion alpha=1.0: AP {delta(fusion10, 'best_AP'):+.3f}, AP_S {delta(fusion10, 'best_AP_S'):+.3f}, AP_L {delta(fusion10, 'best_AP_L'):+.3f}.",
            f"- Fusion alpha=0.5 local: AP {delta(fusion05, 'best_AP'):+.3f}, AP_S {delta(fusion05, 'best_AP_S'):+.3f}, AP_L {delta(fusion05, 'best_AP_L'):+.3f}.",
            f"- Stage shallow: AP {delta(stage, 'best_AP'):+.3f}, AP_S {delta(stage, 'best_AP_S'):+.3f}, AP_L {delta(stage, 'best_AP_L'):+.3f}.",
            "",
            "Interpretation: the VisDrone branch does not yet justify a stronger claim. The only consistent gain is the local alpha=0.5 point on AP/AP_S, but it still has no Fat-side confirmation and AP_L remains weaker. The resumed stage-shallow branch is now clearly negative evidence.",
            "",
            "Next decisive run: AI-TOD-v2 baseline/final on local + Fat, then refresh figures and write-up.",
        ]
    )
    (ANALYSIS_DIR / "analysis_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    summary, curves = build_summary()
    save_csv(ANALYSIS_DIR / "current_results_summary.csv", summary)
    save_csv(ANALYSIS_DIR / "validation_curves.csv", curves)
    plot_current(summary, curves)
    write_notes(summary)
    print(f"Wrote {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
