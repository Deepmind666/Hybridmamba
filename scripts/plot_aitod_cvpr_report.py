#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
OUT_DIR = ROOT / "artifacts" / "analysis" / "aitod_cvpr_20260502_1100"
WATCH_LOG = ROOT / "artifacts" / "queues" / "aitod_dual_watch_20260502_0924" / "watch.log"
FAT_REMOTE_RUN_DIR = r"C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\runs\fat_aitodv2_hybridmambadet_fusion05_resume_mem14_20260502_0923"
FAT_REMOTE_F05_INITIAL_RUN_DIR = r"C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\runs\fat_aitodv2_hybridmambadet_fusion05_mem14_20260502_0133"
FAT_REMOTE_STABLE_RUN_DIR = r"C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\runs\fat_aitodv2_hybridmambadet_stable_mem92_20260501_002325"
DISPLAY_NAME = {
    "TinyViM-B": "TinyViM-B",
    "TinyViM-R": "TinyViM-R",
    "Base": "HybridMamba-Base",
    "Fat-stable": "HybridMambaDet",
    "Fat-f05": "HybridMambaDet-f05",
}
COLOR_MAP = {
    "TinyViM-B": "#000000",
    "TinyViM-R": "#999999",
    "Base": "#228833",
    "Fat-stable": "#CC3311",
    "Fat-f05": "#0077BB",
}

VAL_RE = re.compile(
    r"Epoch\(val\)\s+\[(?P<epoch>\d+)\]\[\s*\d+/\d+\]\s+"
    r"coco/bbox_mAP:\s+(?P<ap>[0-9.]+)\s+"
    r"coco/bbox_mAP_50:\s+(?P<ap50>[0-9.]+)\s+"
    r"coco/bbox_mAP_75:\s+(?P<ap75>[0-9.]+)\s+"
    r"coco/bbox_mAP_s:\s+(?P<aps>[0-9.]+)\s+"
    r"coco/bbox_mAP_m:\s+(?P<apm>[0-9.]+)\s+"
    r"coco/bbox_mAP_l:\s+(?P<apl>-?[0-9.]+)"
)

TRAIN_RE = re.compile(
    r"Epoch\(train\)\s+\[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\].*?"
    r"eta:\s+(?P<eta>.*?)(?=\s+time:)"
)


@dataclass(frozen=True)
class RunSummary:
    label: str
    color: str
    best_ap: float
    latest_ap: float
    best_epoch: int
    latest_epoch: int
    note: str = ""


@dataclass(frozen=True)
class ProgressRow:
    label: str
    color: str
    epoch: int
    current_iter: int
    total_iter: int
    eta: str


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_val_points_text(text: str) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for match in VAL_RE.finditer(text):
        points.append(
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
    points.sort(key=lambda row: row["epoch"])
    return points


def read_remote_text(path: str, tail: int | None = None) -> str:
    tail_part = f" -Tail {tail}" if tail is not None else ""
    command = (
        "powershell -NoProfile -Command "
        + f"\"Get-Content -LiteralPath '{path}'{tail_part}\""
    )
    try:
        return subprocess.check_output(["ssh", "FatMachine", command], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def read_remote_json(path: str) -> dict[str, Any]:
    text = read_remote_text(path)
    if not text.strip():
        return {}
    return json.loads(text)


def parse_val_points(path: Path) -> list[dict[str, float]]:
    return parse_val_points_text(read_text(path))


def parse_last_train(path: Path) -> ProgressRow | None:
    return parse_last_train_text(read_text(path), path.parent.name)


def parse_last_train_text(text: str, source_name: str, label_override: str | None = None, color_override: str | None = None) -> ProgressRow | None:
    matches = list(TRAIN_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    label = label_override or source_name
    if color_override is not None:
        color = color_override
    elif "local_aitodv2_hybridmamba_base_control_resume" in source_name:
        label = "Local base control"
        color = "#1B7F5A"
    elif "fat_aitodv2_hybridmambadet_fusion05_resume" in source_name or label.startswith("Fat"):
        label = "Fat fusion05"
        color = "#D07A1F"
    else:
        color = "#666666"
    return ProgressRow(
        label=label,
        color=color,
        epoch=int(match.group("epoch")),
        current_iter=int(match.group("iter")),
        total_iter=int(match.group("total")),
        eta=match.group("eta").strip(),
    )


def fmt(value: float) -> str:
    return f"{value * 100:.1f}"


def display_name(key: str) -> str:
    return DISPLAY_NAME.get(key, key)


def merge_points(*series_list: list[dict[str, float]]) -> list[dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for series in series_list:
        for row in series:
            merged[int(row["epoch"])] = row
    return [merged[epoch] for epoch in sorted(merged)]


def best_latest_from_curve(curve: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    if not curve:
        fallback = {"epoch": 0, "AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_S": 0.0, "AP_M": 0.0, "AP_L": -1.0}
        return fallback, fallback
    return max(curve, key=lambda row: row["AP"]), curve[-1]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.edgecolor": "#1f1f1f",
            "axes.labelcolor": "#1f1f1f",
            "xtick.color": "#1f1f1f",
            "ytick.color": "#1f1f1f",
            "text.color": "#1f1f1f",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "axes.titlepad": 4.0,
            "figure.dpi": 160,
            "savefig.dpi": 360,
            "font.size": 6.8,
            "axes.titlesize": 7.7,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.6,
            "lines.linewidth": 1.6,
            "lines.markersize": 3.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(0.018, 0.975, letter, transform=ax.transAxes, fontsize=8.4, weight="bold", va="top", ha="left")


def build_summary_rows() -> list[RunSummary]:
    tinyvim_ref = read_json(RUNS / "aitodv2_tinyvim_b_fpn_1x_local_20260422_0142" / "eval_metrics.json")
    tinyvim_retry = read_json(RUNS / "local_aitodv2_tinyvim_stable_retry_mem10_20260501_172729" / "eval_metrics.json")
    local_current = read_json(RUNS / "local_aitodv2_hybridmamba_base_control_resume_mem10_20260502_0923" / "eval_metrics.json")
    local_original = read_json(RUNS / "local_aitodv2_hybridmamba_base_control_mem10_20260502_0129" / "eval_metrics.json")
    fat_stable_curve = parse_val_points_text(read_remote_text(f"{FAT_REMOTE_STABLE_RUN_DIR}\\train.log"))
    fat_stable_best_row, fat_stable_latest_row = best_latest_from_curve(fat_stable_curve)
    fat_initial = parse_val_points_text(read_remote_text(f"{FAT_REMOTE_F05_INITIAL_RUN_DIR}\\train.log"))
    fat_resume = parse_val_points_text(read_remote_text(f"{FAT_REMOTE_RUN_DIR}\\train.log"))
    fat_current_curve = merge_points(fat_initial, fat_resume)
    fat_current_best_row, fat_current_latest_row = best_latest_from_curve(fat_current_curve)

    fat_current_best = fat_current_best_row["AP"]
    fat_current_best_epoch = int(fat_current_best_row["epoch"])
    fat_current_latest = fat_current_latest_row["AP"]
    fat_current_latest_epoch = int(fat_current_latest_row["epoch"])
    fat_hist_best = fat_stable_best_row["AP"]
    fat_hist_latest = fat_stable_latest_row["AP"]
    fat_hist_best_epoch = int(fat_stable_best_row["epoch"])
    fat_hist_latest_epoch = int(fat_stable_latest_row["epoch"])

    return [
        RunSummary(display_name("TinyViM-B"), COLOR_MAP["TinyViM-B"], tinyvim_ref["best"]["coco/bbox_mAP"], tinyvim_ref["latest"]["coco/bbox_mAP"], tinyvim_ref["best_epoch"], tinyvim_ref["epoch"]),
        RunSummary(display_name("TinyViM-R"), COLOR_MAP["TinyViM-R"], tinyvim_retry["best"]["coco/bbox_mAP"], tinyvim_retry["latest"]["coco/bbox_mAP"], tinyvim_retry["best_epoch"], tinyvim_retry["epoch"]),
        RunSummary(display_name("Base"), COLOR_MAP["Base"], local_current["best"]["coco/bbox_mAP"], local_current["latest"]["coco/bbox_mAP"], local_current["best_epoch"], local_current["epoch"]),
        RunSummary(display_name("Fat-stable"), COLOR_MAP["Fat-stable"], fat_hist_best, fat_hist_latest, fat_hist_best_epoch, fat_hist_latest_epoch, "historical HybridMambaDet reference"),
        RunSummary(display_name("Fat-f05"), COLOR_MAP["Fat-f05"], fat_current_best, fat_current_latest, fat_current_best_epoch, fat_current_latest_epoch, "current HybridMambaDet-f05 run"),
    ]


def build_curves() -> dict[str, list[dict[str, float]]]:
    tinyvim_ref = parse_val_points(RUNS / "aitodv2_tinyvim_b_fpn_1x_local_20260422_0142" / "train.log")
    tinyvim_retry = parse_val_points(RUNS / "local_aitodv2_tinyvim_stable_retry_mem10_20260501_172729" / "train.log")
    local_original = parse_val_points(RUNS / "local_aitodv2_hybridmamba_base_control_mem10_20260502_0129" / "train.log")
    local_resume = parse_val_points(RUNS / "local_aitodv2_hybridmamba_base_control_resume_mem10_20260502_0923" / "train.log")

    # Historical Fat checkpoints from the prior stable run, already confirmed in session memory.
    fat_stable = parse_val_points_text(read_remote_text(f"{FAT_REMOTE_STABLE_RUN_DIR}\\train.log"))
    fat_initial = parse_val_points_text(read_remote_text(f"{FAT_REMOTE_F05_INITIAL_RUN_DIR}\\train.log"))
    fat_resume = parse_val_points_text(read_remote_text(f"{FAT_REMOTE_RUN_DIR}\\train.log"))
    fat_f05 = merge_points(fat_initial, fat_resume)

    local_curve = local_original + local_resume
    return {
        "TinyViM-B": tinyvim_ref,
        "Base": local_curve,
        "Fat-stable": fat_stable,
        "Fat-f05": fat_f05,
    }


def build_progress_rows() -> list[ProgressRow]:
    local_progress = parse_last_train(RUNS / "local_aitodv2_hybridmamba_base_control_resume_mem10_20260502_0923" / "train.log")
    fat_log_text = read_remote_text(f"{FAT_REMOTE_RUN_DIR}\\train.log", tail=160)
    fat_progress = parse_last_train_text(fat_log_text, "fat_aitodv2_hybridmambadet_fusion05_resume_mem14_20260502_0923", "Fat fusion05", "#D07A1F")
    rows = []
    if local_progress is not None:
        local_progress = ProgressRow(display_name("Base"), COLOR_MAP["Base"], local_progress.epoch, local_progress.current_iter, local_progress.total_iter, local_progress.eta)
        rows.append(local_progress)
    if fat_progress is not None:
        fat_progress = ProgressRow(display_name("Fat-f05"), COLOR_MAP["Fat-f05"], fat_progress.epoch, fat_progress.current_iter, fat_progress.total_iter, fat_progress.eta)
        rows.append(fat_progress)
    return rows


def plot_endpoint_panel(ax: plt.Axes, rows: list[RunSummary]) -> None:
    order = [display_name("TinyViM-B"), display_name("Fat-stable"), display_name("TinyViM-R"), display_name("Base"), display_name("Fat-f05")]
    y = list(range(len(order)))[::-1]
    lookup = {row.label: row for row in rows}

    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.55)
    ax.set_axisbelow(True)

    for yy, key in zip(y, order):
        row = lookup[key]
        latest = fmt(row.latest_ap)
        best = fmt(row.best_ap)
        ax.plot([row.latest_ap * 100, row.best_ap * 100], [yy, yy], color=row.color, alpha=0.85, linewidth=1.35)
        ax.scatter([row.latest_ap * 100], [yy], s=28, facecolor="white", edgecolor=row.color, linewidth=1.0, zorder=4)
        ax.scatter([row.best_ap * 100], [yy], s=22, color=row.color, edgecolor="white", linewidth=0.4, zorder=5)
        ax.text(row.best_ap * 100 + 0.18, yy + 0.01, best, ha="left", va="center", fontsize=7.0)
        if row.latest_ap != row.best_ap:
            ax.text(row.latest_ap * 100 + 0.18, yy - 0.14, f"latest {latest}", ha="left", va="center", fontsize=6.4, color="#606060")

    ax.set_yticks(y)
    ax.set_yticklabels([display_name("TinyViM-B"), display_name("Fat-stable"), display_name("TinyViM-R"), display_name("Base"), display_name("Fat-f05")])
    ax.set_xlim(0, 18.0)
    ax.set_xlabel("AP points")
    ax.set_title("Endpoint comparison", loc="left")
    panel_letter(ax, "a")


def plot_curve_panel(ax: plt.Axes, curves: dict[str, list[dict[str, float]]]) -> None:
    label_map = {
        "TinyViM-B": display_name("TinyViM-B"),
        "Base": display_name("Base"),
        "Fat-stable": display_name("Fat-stable"),
        "Fat-f05": display_name("Fat-f05"),
    }
    label_offsets = {
        "TinyViM-B": 0.18,
        "Base": 0.16,
        "Fat-stable": -0.18,
        "Fat-f05": 0.16,
    }
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.55)
    ax.set_axisbelow(True)
    for key in ["TinyViM-B", "Base", "Fat-stable", "Fat-f05"]:
        rows = curves[key]
        if not rows:
            continue
        xs = [row["epoch"] for row in rows]
        ys = [row["AP"] * 100 for row in rows]
        color = COLOR_MAP[key]
        linestyle = "-" if key != "Fat-stable" else (0, (3, 2))
        ax.plot(xs, ys, color=color, linestyle=linestyle, marker="o", markersize=3.2, linewidth=1.55, alpha=0.95)
        ax.scatter([xs[-1]], [ys[-1]], s=24, facecolor="white", edgecolor=color, linewidth=0.95, zorder=4)
        ax.text(xs[-1] + 0.25, ys[-1] + label_offsets[key], label_map[key], color=color, fontsize=6.6, va="center")

    ax.set_xlim(0.6, 16.8)
    ax.set_ylim(0, 18.2)
    ax.set_xlabel("Validation epoch")
    ax.set_ylabel("AP points")
    ax.set_title("Validation trajectories", loc="left")
    panel_letter(ax, "b")


def build_metric_triptych_runs() -> dict[str, list[dict[str, float]]]:
    tinyvim_ref = parse_val_points(RUNS / "aitodv2_tinyvim_b_fpn_1x_local_20260422_0142" / "train.log")
    local_original = parse_val_points(RUNS / "local_aitodv2_hybridmamba_base_control_mem10_20260502_0129" / "train.log")
    local_resume = parse_val_points(RUNS / "local_aitodv2_hybridmamba_base_control_resume_mem10_20260502_0923" / "train.log")
    fat_initial = parse_val_points_text(read_remote_text(f"{FAT_REMOTE_F05_INITIAL_RUN_DIR}\\train.log"))
    fat_resume = parse_val_points_text(read_remote_text(f"{FAT_REMOTE_RUN_DIR}\\train.log"))
    return {
        "TinyViM-B": tinyvim_ref,
        "Base": merge_points(local_original, local_resume),
        "Fat-f05": merge_points(fat_initial, fat_resume),
    }


def metric_ylim(runs: dict[str, list[dict[str, float]]], metric_key: str) -> tuple[float, float]:
    values = [
        row[metric_key] * 100
        for rows in runs.values()
        for row in rows
        if metric_key in row and row[metric_key] >= 0
    ]
    if not values:
        return (0.0, 1.0)
    raw_upper = max(values) * 1.14
    step = 1.0 if raw_upper <= 10 else 2.5
    return (0.0, math.ceil(raw_upper / step) * step)


def plot_metric_panel(
    ax: plt.Axes,
    metric_key: str,
    title: str,
    runs: dict[str, list[dict[str, float]]],
) -> None:
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)

    for key in ["TinyViM-B", "Base", "Fat-stable", "Fat-f05"]:
        rows = runs.get(key, [])
        if not rows:
            continue
        xs = [row["epoch"] for row in rows]
        ys = [row[metric_key] * 100 for row in rows]
        color = COLOR_MAP[key]
        ax.plot(
            xs,
            ys,
            color=color,
            linestyle="-",
            linewidth=3.2,
            alpha=0.09,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            xs,
            ys,
            color=color,
            linestyle="-",
            marker="o",
            markersize=2.9,
            linewidth=1.65,
            alpha=0.96,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter([xs[-1]], [ys[-1]], s=24, facecolor="white", edgecolor=color, linewidth=0.95, zorder=4, clip_on=False)

    ax.set_xlim(0.6, 16.8)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_ylim(*metric_ylim(runs, metric_key))
    ax.set_xlabel("Validation epoch")
    ax.set_ylabel("AP points")
    ax.set_title(title, loc="left")


def plot_metric_triptych() -> None:
    runs = build_metric_triptych_runs()
    stable = build_curves().get("Fat-stable", [])
    if stable:
        runs["Fat-stable"] = stable
    fig = plt.figure(figsize=(7.25, 2.45))
    gs = fig.add_gridspec(1, 3, left=0.065, right=0.992, bottom=0.27, top=0.86, wspace=0.28)
    metrics = [
        ("AP", "AP"),
        ("AP50", "AP50"),
        ("AP75", "AP75"),
    ]
    for idx, (metric_key, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        plot_metric_panel(ax, metric_key, title, runs)
        panel_letter(ax, chr(ord("a") + idx))

    handles = [
        Line2D([0], [0], color=COLOR_MAP["TinyViM-B"], marker="o", linewidth=1.65, markersize=3.2, label=display_name("TinyViM-B")),
        Line2D([0], [0], color=COLOR_MAP["Base"], marker="o", linewidth=1.65, markersize=3.2, label=display_name("Base")),
        Line2D([0], [0], color=COLOR_MAP["Fat-stable"], marker="o", linewidth=1.65, markersize=3.2, label=display_name("Fat-stable")),
        Line2D([0], [0], color=COLOR_MAP["Fat-f05"], marker="o", linewidth=1.65, markersize=3.2, label=display_name("Fat-f05")),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.045), handlelength=1.7, columnspacing=1.3)
    save_figure(fig, "figure_aitod_cvpr_metric_triptych")
    plt.close(fig)


def plot_progress_panel(ax: plt.Axes, rows: list[ProgressRow]) -> None:
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_yticks(list(range(len(rows)))[::-1])
    ax.set_yticklabels([row.label for row in rows])
    ax.set_xlabel("Epoch completion")
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.55)
    ax.set_axisbelow(True)

    for yi, row in zip(list(range(len(rows)))[::-1], rows):
        progress = row.current_iter / max(row.total_iter, 1)
        ax.barh(yi, 1.0, height=0.34, color="#EEF0F2", edgecolor="#D8DDE2", linewidth=0.6)
        ax.barh(yi, progress, height=0.34, color=row.color, edgecolor="none", alpha=0.95)
        ax.text(1.02, yi + 0.09, f"E{row.epoch}  {row.current_iter}/{row.total_iter}", ha="left", va="center", fontsize=7.0)
        ax.text(1.02, yi - 0.12, f"{progress * 100:.1f}%  |  eta {row.eta}", ha="left", va="center", fontsize=6.3, color="#5f5f5f")

    ax.set_title("Live progress", loc="left")
    panel_letter(ax, "c")


def write_notes(rows: list[RunSummary], progress_rows: list[ProgressRow]) -> None:
    with (OUT_DIR / "aitod_cvpr_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", "best_ap_points", "latest_ap_points", "best_epoch", "latest_epoch", "note"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run": row.label,
                    "best_ap_points": f"{fmt(row.best_ap)}",
                    "latest_ap_points": f"{fmt(row.latest_ap)}",
                    "best_epoch": row.best_epoch,
                    "latest_epoch": row.latest_epoch,
                    "note": row.note,
                }
            )

    lines = [
        "# AI-TOD-v2 CV-style stage snapshot",
        "",
        "| Run | Best AP | Latest AP | Best epoch | Latest epoch | Note |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| {row.label} | {fmt(row.best_ap)} | {fmt(row.latest_ap)} | {row.best_epoch} | {row.latest_epoch} | {row.note} |")
    lines += [
        "",
        "## Live progress",
    ]
    for row in progress_rows:
        lines.append(f"- {row.label}: epoch {row.epoch}, iter {row.current_iter}/{row.total_iter}, eta {row.eta}.")
    (OUT_DIR / "aitod_cvpr_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    rows = build_summary_rows()
    curves = build_curves()
    progress_rows = build_progress_rows()

    fig = plt.figure(figsize=(7.6, 3.55))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.985, bottom=0.18, top=0.93, wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    plot_endpoint_panel(ax0, rows)
    plot_curve_panel(ax1, curves)
    fig.text(0.5, 0.055, "AI-TOD-v2, RetinaNet + FPN, COCO AP points", ha="center", va="center", fontsize=6.8, color="#666666")
    save_figure(fig, "figure_aitod_cvpr_summary")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 1.95))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.22)
    plot_progress_panel(ax, progress_rows)
    fig.text(0.09, 0.06, "Resumed dual-machine run; next validation is pending.", ha="left", va="center", fontsize=6.7, color="#666666")
    save_figure(fig, "figure_aitod_cvpr_status")
    plt.close(fig)

    plot_metric_triptych()

    write_notes(rows, progress_rows)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
