#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import ticker

from publication_style import apply_publication_style, MUTED, TEXT


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
QUEUE = ROOT / "artifacts" / "queues"
OUT_DIR = ROOT / "artifacts" / "analysis" / "aitod_current_20260502_0950"
WATCH_LOG = QUEUE / "aitod_dual_watch_20260502_0924" / "watch.log"

VAL_RE = re.compile(
    r"Epoch\(val\)\s+\[(?P<epoch>\d+)\]\[\s*\d+/\d+\]\s+"
    r"coco/bbox_mAP:\s+(?P<ap>-?[0-9.]+)\s+"
    r"coco/bbox_mAP_50:\s+(?P<ap50>-?[0-9.]+)\s+"
    r"coco/bbox_mAP_75:\s+(?P<ap75>-?[0-9.]+)\s+"
    r"coco/bbox_mAP_s:\s+(?P<aps>-?[0-9.]+)\s+"
    r"coco/bbox_mAP_m:\s+(?P<apm>-?[0-9.]+)\s+"
    r"coco/bbox_mAP_l:\s+(?P<apl>-?[0-9.]+)"
)

TRAIN_RE = re.compile(
    r"Epoch\(train\)\s+\[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\].*?"
    r"eta:\s+(?P<eta>.*?)(?=\s+time:)"
)

DISPLAY_LABELS = {
    "TinyViM-B 1x": "TinyViM",
    "TinyViM retry": "TinyViM retry",
    "HybridMamba Base": "Base",
    "HybridMambaDet stable": "Fat stable",
    "HybridMambaDet fusion05": "Fat fusion05",
}


@dataclass(frozen=True)
class RunMetrics:
    label: str
    color: str
    best_ap: float | None
    latest_ap: float | None
    best_epoch: int | None
    latest_epoch: int | None
    note: str = ""
    source: str = ""


@dataclass(frozen=True)
class ProgressSnapshot:
    label: str
    color: str
    epoch: int
    iter_in_epoch: int
    total_iter: int
    source: str
    note: str = ""


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_val_curve(log_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for match in VAL_RE.finditer(read_text(log_path)):
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
    rows.sort(key=lambda row: row["epoch"])
    return rows


def parse_train_snapshot(log_path: Path) -> ProgressSnapshot | None:
    matches = list(TRAIN_RE.finditer(read_text(log_path)))
    if not matches:
        return None
    match = matches[-1]
    return ProgressSnapshot(
        label=log_path.parent.name,
        color="#777777",
        epoch=int(match.group("epoch")),
        iter_in_epoch=int(match.group("iter")),
        total_iter=int(match.group("total")),
        source=str(log_path),
        note=match.group("eta").strip(),
    )


def parse_watch_snapshot(watch_log: Path, run_id: str, label: str, color: str) -> ProgressSnapshot | None:
    text = read_text(watch_log)
    pattern = re.compile(
        rf"\[(?:Local|Fat)\] RunId: {re.escape(run_id)}\n(?P<body>.*?)(?=\n\[(?:Local|Fat)\] RunId: |\n=== Local \+ Fat watcher ===|\Z)",
        re.S,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    body = matches[-1].group("body")
    train_matches = list(TRAIN_RE.finditer(body))
    if not train_matches:
        return None
    last = train_matches[-1]
    return ProgressSnapshot(
        label=label,
        color=color,
        epoch=int(last.group("epoch")),
        iter_in_epoch=int(last.group("iter")),
        total_iter=int(last.group("total")),
        source=str(watch_log),
        note=last.group("eta").strip(),
    )


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.{digits}f}"


def ap_points(value: float | None) -> float:
    return 0.0 if value is None else value * 100.0


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")


def setup_style() -> None:
    apply_publication_style()
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 450,
            "font.size": 8.1,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=10.3, weight="bold", va="top")


def load_run_summary() -> tuple[list[RunMetrics], dict[str, list[dict[str, float]]], dict[str, ProgressSnapshot]]:
    tinyvim_ref_dir = RUNS / "aitodv2_tinyvim_b_fpn_1x_local_20260422_0142"
    tinyvim_retry_dir = RUNS / "local_aitodv2_tinyvim_stable_retry_mem10_20260501_172729"
    base_control_dir = RUNS / "local_aitodv2_hybridmamba_base_control_mem10_20260502_0129"
    base_control_resume_dir = RUNS / "local_aitodv2_hybridmamba_base_control_resume_mem10_20260502_0923"

    tinyvim_curve = parse_val_curve(tinyvim_ref_dir / "train.log")
    retry_curve = parse_val_curve(tinyvim_retry_dir / "train.log")
    base_curve = parse_val_curve(base_control_dir / "train.log")

    # Manual summaries for Fat runs come from the live watcher / prior remote logs.
    fat_stable_curve = [
        {"epoch": 1, "AP": 0.024, "AP50": 0.064, "AP75": 0.011, "AP_S": 0.023, "AP_M": 0.047, "AP_L": -1.000},
        {"epoch": 6, "AP": 0.073, "AP50": 0.183, "AP75": 0.042, "AP_S": 0.073, "AP_M": 0.119, "AP_L": -1.000},
        {"epoch": 11, "AP": 0.081, "AP50": 0.201, "AP75": 0.047, "AP_S": 0.080, "AP_M": 0.129, "AP_L": -1.000},
        {"epoch": 16, "AP": 0.084, "AP50": 0.208, "AP75": 0.050, "AP_S": 0.083, "AP_M": 0.133, "AP_L": -1.000},
    ]
    fat_current_curve = [
        {"epoch": 1, "AP": 0.024, "AP50": 0.064, "AP75": 0.011, "AP_S": 0.023, "AP_M": 0.047, "AP_L": -1.000},
        {"epoch": 3, "AP": 0.060, "AP50": 0.150, "AP75": 0.036, "AP_S": 0.060, "AP_M": 0.092, "AP_L": -1.000},
    ]

    runs = [
        RunMetrics(
            label="TinyViM-B 1x",
            color="#1F5AA6",
            best_ap=0.166,
            latest_ap=0.166,
            best_epoch=11,
            latest_epoch=12,
            note="reference baseline",
            source=str(tinyvim_ref_dir),
        ),
        RunMetrics(
            label="TinyViM retry",
            color="#707781",
            best_ap=0.072,
            latest_ap=0.051,
            best_epoch=3,
            latest_epoch=4,
            note="local retry baseline",
            source=str(tinyvim_retry_dir),
        ),
        RunMetrics(
            label="HybridMamba Base",
            color="#0E7C59",
            best_ap=0.052,
            latest_ap=0.052,
            best_epoch=3,
            latest_epoch=3,
            note="initial control",
            source=str(base_control_dir),
        ),
        RunMetrics(
            label="HybridMambaDet stable",
            color="#C23B22",
            best_ap=0.084,
            latest_ap=0.077,
            best_epoch=16,
            latest_epoch=16,
            note="best remote Fat checkpoint",
            source="remote summary",
        ),
        RunMetrics(
            label="HybridMambaDet fusion05",
            color="#D67A1F",
            best_ap=0.060,
            latest_ap=0.060,
            best_epoch=3,
            latest_epoch=3,
            note="current Fat snapshot",
            source="remote summary",
        ),
    ]

    summary_curves = {
        "TinyViM-B 1x": tinyvim_curve,
        "TinyViM retry": retry_curve,
        "HybridMamba Base": base_curve,
        "HybridMambaDet stable": fat_stable_curve,
        "HybridMambaDet fusion05": fat_current_curve,
    }

    live = {
        "local": parse_train_snapshot(base_control_resume_dir / "train.log"),
        "fat": parse_watch_snapshot(
            WATCH_LOG,
            "fat_aitodv2_hybridmambadet_fusion05_resume_mem14_20260502_0923",
            "Fat fusion05",
            "#D67A1F",
        ),
    }
    if live["local"] is not None:
        live["local"] = ProgressSnapshot(
            label="Local base control",
            color="#0E7C59",
            epoch=live["local"].epoch,
            iter_in_epoch=live["local"].iter_in_epoch,
            total_iter=live["local"].total_iter,
            source=live["local"].source,
            note=live["local"].note,
        )

    return runs, summary_curves, live


def plot_summary(ax: plt.Axes, runs: list[RunMetrics]) -> None:
    y = list(range(len(runs)))[::-1]
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.55)
    ax.set_axisbelow(True)

    for yi, run in zip(y, runs):
        label = DISPLAY_LABELS.get(run.label, run.label)
        best = ap_points(run.best_ap)
        latest = ap_points(run.latest_ap)
        ax.barh(yi, best, height=0.44, color=run.color, alpha=0.92, edgecolor="none")
        ax.scatter([latest], [yi], s=28, facecolor="white", edgecolor=run.color, linewidth=1.0, zorder=4)
        if run.latest_ap is not None and run.best_ap is not None and abs(run.latest_ap - run.best_ap) > 1e-6:
            ax.plot([latest, best], [yi, yi], color=run.color, linewidth=1.0, alpha=0.75)
        ax.text(best + 0.18, yi + 0.01, f"{best:.1f}", ha="left", va="center", fontsize=7.2)

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_LABELS.get(run.label, run.label) for run in runs])
    ax.set_xlim(0, 17.2)
    ax.set_xlabel("AP points")
    ax.set_title("Best validation AP snapshot", loc="left", pad=6)
    panel_letter(ax, "a")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_trajectories(ax: plt.Axes, curves: dict[str, list[dict[str, float]]], runs: list[RunMetrics]) -> None:
    label_to_run = {run.label: run for run in runs}
    order = ["TinyViM-B 1x", "TinyViM retry", "HybridMamba Base", "HybridMambaDet stable", "HybridMambaDet fusion05"]
    for label in order:
        rows = curves[label]
        if not rows:
            continue
        xs = [row["epoch"] for row in rows]
        ys = [row["AP"] * 100.0 for row in rows]
        run = label_to_run[label]
        linestyle = "-" if label != "HybridMambaDet stable" else (0, (3, 2))
        ax.plot(
            xs,
            ys,
            color=run.color,
            linewidth=1.7,
            linestyle=linestyle,
            solid_capstyle="round",
            label=DISPLAY_LABELS.get(label, label),
        )
        ax.scatter(xs, ys, color=run.color, s=14, edgecolor="white", linewidth=0.35, zorder=4)
        best_epoch = run.best_epoch
        if best_epoch is not None:
            best_match = next((row for row in rows if int(row["epoch"]) == best_epoch), None)
            if best_match is not None:
                ax.scatter(
                    [best_match["epoch"]],
                    [best_match["AP"] * 100.0],
                    s=40,
                    facecolor="white",
                    edgecolor=run.color,
                    linewidth=1.0,
                    zorder=5,
                )

    ax.set_xlabel("Validation epoch")
    ax.set_ylabel("AP points")
    ax.set_title("Validation trajectories", loc="left", pad=6)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax.set_xlim(0.5, 17.3)
    ax.set_ylim(0.0, 17.8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=False))
    ax.legend(frameon=False, loc="lower right", ncols=1, handlelength=1.8, handletextpad=0.45, borderpad=0.1)
    panel_letter(ax, "b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_live_progress(ax: plt.Axes, live: dict[str, ProgressSnapshot | None]) -> None:
    rows = [item for item in [live["local"], live["fat"]] if item is not None]
    y = list(range(len(rows)))[::-1]
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([row.label for row in rows])
    ax.set_xlabel("Current epoch completion")
    ax.set_title("Live dual-machine progress", loc="left", pad=6)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.55)
    ax.set_axisbelow(True)

    for yi, row in zip(y, rows):
        progress = row.iter_in_epoch / max(row.total_iter, 1)
        ax.barh(yi, 1.0, height=0.42, color="#EEF1F3", edgecolor="#D0D5DA", linewidth=0.6)
        ax.barh(yi, progress, height=0.42, color=row.color, edgecolor="none", alpha=0.92)
        ax.text(1.02, yi + 0.08, f"E{row.epoch}  {row.iter_in_epoch}/{row.total_iter}", ha="left", va="center", fontsize=7.2)
        ax.text(1.02, yi - 0.14, f"{progress * 100.0:.1f}% of epoch 4  |  post-restart val pending", ha="left", va="center", fontsize=6.4, color=MUTED)

    panel_letter(ax, "c")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_metrics_table(path_csv: Path, path_md: Path, runs: list[RunMetrics]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.append(
            {
                "run": run.label,
                "best_ap_points": f"{ap_points(run.best_ap):.1f}",
                "latest_ap_points": f"{ap_points(run.latest_ap):.1f}",
                "best_epoch": run.best_epoch,
                "latest_epoch": run.latest_epoch,
                "note": run.note,
                "source": run.source,
            }
        )
    with path_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# AI-TOD-v2 stage snapshot",
        "",
        "| Run | Best AP | Latest AP | Best epoch | Latest epoch | Note |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['best_ap_points']} | {row['latest_ap_points']} | {row['best_epoch']} | {row['latest_epoch']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Live progress",
            "- Local resumed base control: epoch 4, validation pending after restart.",
            "- Fat resumed fusion05: epoch 4, validation pending after restart.",
            "",
            "## Reading",
            "- TinyViM-B 1x remains the strong benchmark at 16.6 AP.",
            "- The current Mamba branch is still below the benchmark on AI-TOD-v2.",
            "- Current live runs are still moving and have not crashed after the reboot.",
        ]
    )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary_figure(runs: list[RunMetrics], curves: dict[str, list[dict[str, float]]], live: dict[str, ProgressSnapshot | None]) -> None:
    fig = plt.figure(figsize=(12.4, 4.9), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.08, 0.92),
        left=0.065,
        right=0.985,
        top=0.82,
        bottom=0.14,
        wspace=0.30,
    )

    fig.text(0.065, 0.95, "AI-TOD-v2 HybridMamba stage report snapshot", ha="left", va="top", fontsize=10.8, weight="bold")
    fig.text(
        0.065,
        0.905,
        "Best checkpoints and live dual-machine progress. Values are COCO AP points; the resumed pair is still in epoch 4 with no new post-restart validation yet.",
        ha="left",
        va="top",
        fontsize=7.7,
        color=MUTED,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    plot_summary(ax_a, runs)
    plot_trajectories(ax_b, curves, runs)
    plot_live_progress(ax_b, live)

    save_figure(fig, "figure_aitod_stage_summary")
    plt.close(fig)


def plot_live_figure(live: dict[str, ProgressSnapshot | None]) -> None:
    rows = [item for item in [live["local"], live["fat"]] if item is not None]
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    fig.text(0.065, 0.96, "Current dual-machine training progress", ha="left", va="top", fontsize=10.6, weight="bold")
    fig.text(
        0.065,
        0.90,
        "Both runs are still in epoch 4. The next validation is expected after the current epoch ends.",
        ha="left",
        va="top",
        fontsize=7.4,
        color=MUTED,
    )

    y = list(range(len(rows)))[::-1]
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([row.label for row in rows])
    ax.set_xlabel("Current epoch completion")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.55)
    ax.set_axisbelow(True)

    for yi, row in zip(y, rows):
        progress = row.iter_in_epoch / max(row.total_iter, 1)
        ax.barh(yi, 1.0, height=0.40, color="#EEF1F3", edgecolor="#D0D5DA", linewidth=0.6)
        ax.barh(yi, progress, height=0.40, color=row.color, edgecolor="none", alpha=0.92)
        ax.text(1.02, yi + 0.08, f"E{row.epoch}  {row.iter_in_epoch}/{row.total_iter}", ha="left", va="center", fontsize=7.1)
        ax.text(1.02, yi - 0.15, f"{progress * 100.0:.1f}% of epoch", ha="left", va="center", fontsize=6.5, color=MUTED)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, "figure_aitod_live_progress")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    runs, curves, live = load_run_summary()
    write_metrics_table(OUT_DIR / "aitod_stage_metrics.csv", OUT_DIR / "aitod_stage_metrics.md", runs)
    plot_summary_figure(runs, curves, live)
    plot_live_figure(live)
    print(f"Wrote figures and notes to {OUT_DIR}")


if __name__ == "__main__":
    main()
