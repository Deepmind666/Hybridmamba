from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\mamba")
OUT = ROOT / "artifacts" / "analysis" / "aitod_current_20260504_1021"
RAW = OUT / "raw"


METHODS = {
    "local_aitodv2_tinyvim_b_72e_sameprotocol_mem8_20260502_2154": "TinyViM-B + RetinaNet",
    "local_aitodv2_hybridmamba_base_control_mem10_20260502_0129": "HybridMamba-B + RetinaNet",
    "local_aitodv2_hybridmamba_base_control_resume_mem10_20260502_0923": "HybridMamba-B + RetinaNet",
    "local_aitodv2_hybridmamba_base_72e_resume_mem10_20260502_155411": "HybridMamba-B + RetinaNet",
    "local_aitodv2_hybridmamba_base_72e_resume_mem8_recover1_20260502_2018": "HybridMamba-B + RetinaNet",
    "fat_aitodv2_hybridmambadet_stable_mem92_20260501_002325": "HybridMambaDet-B + RetinaNet",
    "fat_aitodv2_hybridmambadet_fusion05_mem14_20260502_0133": "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5",
    "fat_aitodv2_hybridmambadet_fusion05_resume_mem14_20260502_0923": "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5",
    "fat_aitodv2_hybridmambadet_fusion05_72e_resume_mem14_repair1_20260502_1840": "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5",
    "fat_aitodv2_hybridmambadet_fusion05_72e_resume_mem14_repair1_correctresume_cmd_20260503_1605": "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5",
}


DISPLAY = {
    "TinyViM-B + RetinaNet": "TinyViM-B + RetinaNet",
    "HybridMamba-B + RetinaNet": "HybridMamba-B + RetinaNet",
    "HybridMambaDet-B + RetinaNet": "HybridMambaDet-B + RetinaNet",
    "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5": "HybridMambaDet-B + RetinaNet (alpha=0.5)",
}


COLORS = {
    "TinyViM-B + RetinaNet": "#0072B2",
    "HybridMamba-B + RetinaNet": "#009E73",
    "HybridMambaDet-B + RetinaNet": "#E69F00",
    "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5": "#D55E00",
}


VAL_RE = re.compile(
    r"Epoch\(val\)\s+\[(?P<epoch>\d+)\]\[\d+/\d+\]\s+"
    r"coco/bbox_mAP:\s+(?P<ap>-?\d+\.\d+)\s+"
    r"coco/bbox_mAP_50:\s+(?P<ap50>-?\d+\.\d+)\s+"
    r"coco/bbox_mAP_75:\s+(?P<ap75>-?\d+\.\d+)\s+"
    r"coco/bbox_mAP_s:\s+(?P<aps>-?\d+\.\d+)\s+"
    r"coco/bbox_mAP_m:\s+(?P<apm>-?\d+\.\d+)"
)


def method_from_run(run: str) -> str | None:
    run_name = Path(run).name
    return METHODS.get(run_name)


def parse_raw(path: Path) -> list[dict]:
    rows: list[dict] = []
    current_run = None
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        raw = raw.lstrip("\ufeff")
        if raw.startswith("RUN="):
            current_run = raw[4:].strip()
            continue
        if current_run is None or "Epoch(val)" not in raw:
            continue
        match = VAL_RE.search(raw)
        if not match:
            continue
        method = method_from_run(current_run)
        if method is None:
            continue
        rows.append(
            {
                "run": current_run,
                "method": method,
                "epoch": int(match.group("epoch")),
                "AP": float(match.group("ap")),
                "AP50": float(match.group("ap50")),
                "AP75": float(match.group("ap75")),
                "AP_S": float(match.group("aps")),
                "AP_M": float(match.group("apm")),
            }
        )
    return rows


def add_manual_rows(rows: list[dict]) -> None:
    # The repair1 run only kept eval_metrics.json and checkpoints for epochs 12-13.
    # These records are copied from that run's eval_metrics.json and checkpoint metadata.
    run = (
        r"C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\runs"
        r"\fat_aitodv2_hybridmambadet_fusion05_72e_resume_mem14_repair1_20260502_1840"
    )
    method = "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5"
    rows.extend(
        [
            {
                "run": run,
                "method": method,
                "epoch": 12,
                "AP": 0.074,
                "AP50": 0.169,
                "AP75": 0.052,
                "AP_S": 0.072,
                "AP_M": 0.156,
            },
            {
                "run": run,
                "method": method,
                "epoch": 13,
                "AP": 0.071,
                "AP50": 0.168,
                "AP75": 0.046,
                "AP_S": 0.071,
                "AP_M": 0.120,
            },
        ]
    )


def build_dataframe() -> pd.DataFrame:
    rows = []
    rows.extend(parse_raw(RAW / "local_metric_lines.txt"))
    rows.extend(parse_raw(RAW / "fat_metric_lines.txt"))
    add_manual_rows(rows)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No validation metrics parsed")
    df = (
        df.sort_values(["method", "epoch", "run"])
        .drop_duplicates(["method", "epoch"], keep="last")
        .sort_values(["method", "epoch"])
        .reset_index(drop=True)
    )
    return df


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def format_axis(ax, ylabel: str, xlabel: bool = True) -> None:
    ax.set_xlabel("Validation epoch" if xlabel else "")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.5, 16.5)
    ax.set_xticks([1, 4, 8, 12, 16])


def plot_trajectory(df: pd.DataFrame) -> None:
    metrics = [("AP", "AP"), ("AP50", "AP50"), ("AP75", "AP75"), ("AP_S", "AP small")]
    order = [
        "TinyViM-B + RetinaNet",
        "HybridMamba-B + RetinaNet",
        "HybridMambaDet-B + RetinaNet",
        "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5",
    ]
    fig = plt.figure(figsize=(8.9, 5.9))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.10, 1, 1], hspace=0.34, wspace=0.22)
    top_ax = fig.add_subplot(gs[0, :])
    top_ax.axis("off")
    axes = np.array(
        [
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
            [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
        ]
    )
    for panel_idx, (ax, (metric, label)) in enumerate(zip(axes.flat, metrics)):
        ymax = max(0.11, df[metric].max() * 1.18)
        for method in order:
            part = df[df["method"] == method].sort_values("epoch")
            if part.empty:
                continue
            ax.plot(
                part["epoch"],
                part[metric],
                color=COLORS[method],
                lw=1.8,
                marker="o",
                ms=3.6,
                mec="white",
                mew=0.7,
                label=DISPLAY[method],
                alpha=0.98,
            )
            best_idx = part[metric].idxmax()
            best = part.loc[best_idx]
            ax.scatter(
                [best["epoch"]],
                [best[metric]],
                s=54,
                facecolor="white",
                edgecolor=COLORS[method],
                linewidth=1.5,
                zorder=5,
            )
        ax.set_ylim(0, ymax)
        format_axis(ax, label, xlabel=panel_idx >= 2)
        ax.text(
            0.015,
            0.96,
            chr(ord("a") + panel_idx),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=9,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    top_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.085, top=0.965)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"figure_aitod_cvpr_trajectory.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_gap(df: pd.DataFrame) -> None:
    best = (
        df.sort_values("AP")
        .groupby("method", as_index=False)
        .tail(1)
        .sort_values("AP", ascending=False)
    )
    latest = df.sort_values("epoch").groupby("method", as_index=False).tail(1)
    summary = best.merge(latest[["method", "epoch", "AP", "AP50", "AP75", "AP_S"]], on="method", suffixes=("_best", "_latest"))
    order = [
        "TinyViM-B + RetinaNet",
        "HybridMambaDet-B + RetinaNet",
        "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5",
        "HybridMamba-B + RetinaNet",
    ]
    summary["order"] = summary["method"].map({m: i for i, m in enumerate(order)})
    summary = summary.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65), gridspec_kw={"width_ratios": [1.15, 1.0]}, constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(summary))
    width = 0.18
    metrics = [("AP", "AP"), ("AP50", "AP50"), ("AP75", "AP75"), ("AP_S", "AP_S")]
    palette = ["#0072B2", "#56B4E9", "#D55E00", "#009E73"]
    for i, (metric, label) in enumerate(metrics):
        vals = summary[f"{metric}_best"].to_numpy()
        ax.bar(x + (i - 1.5) * width, vals, width=width, color=palette[i], label=label, edgecolor="black", linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m].replace(" + RetinaNet", "") for m in summary["method"]], rotation=18, ha="right")
    ax.set_ylabel("Best validation score")
    ax.grid(axis="y", alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(-0.01, 1.12), columnspacing=0.9)
    ax.set_ylim(0, max(summary["AP50_best"].max() * 1.16, 0.27))

    ref_method = "TinyViM-B + RetinaNet"
    ref_ap = float(summary.loc[summary["method"] == ref_method, "AP_best"].iloc[0])
    gap = summary.copy()
    gap["AP_gap_vs_TinyViM"] = gap["AP_best"] - ref_ap
    ax2 = axes[1]
    colors = [COLORS[m] for m in gap["method"]]
    ax2.barh(np.arange(len(gap)), gap["AP_gap_vs_TinyViM"], color=colors, edgecolor="black", linewidth=0.4)
    ax2.axvline(0, color="#222222", lw=0.9)
    ax2.set_yticks(np.arange(len(gap)))
    ax2.set_yticklabels([DISPLAY[m].replace(" + RetinaNet", "") for m in gap["method"]])
    ax2.set_xlabel("Best AP difference vs TinyViM-B")
    ax2.grid(axis="x", alpha=0.75)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for idx, val in enumerate(gap["AP_gap_vs_TinyViM"]):
        ax2.text(val - 0.003 if val < 0 else val + 0.003, idx, f"{val:+.3f}", va="center", ha="right" if val < 0 else "left", fontsize=8)
    fig.suptitle("Best observed AI-TOD-v2 evidence favors the TinyViM-B baseline", x=0.02, y=1.03, ha="left", fontweight="bold")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"figure_aitod_cvpr_best_gap.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(OUT / "aitod_cvpr_best_summary.csv", index=False)


def plot_decision_panel(df: pd.DataFrame) -> None:
    order = [
        "TinyViM-B + RetinaNet",
        "HybridMambaDet-B + RetinaNet",
        "HybridMambaDet-B + RetinaNet, detail fusion alpha=0.5",
        "HybridMamba-B + RetinaNet",
    ]
    metrics = ["AP", "AP50", "AP75", "AP_S"]
    best = df.sort_values("AP").groupby("method", as_index=False).tail(1)
    best = best.set_index("method").reindex(order).dropna(how="all").reset_index()
    mat = best[metrics].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.1, 3.2), constrained_layout=True)
    im = ax.imshow(mat, cmap="cividis", aspect="auto", vmin=0, vmax=max(0.26, float(np.nanmax(mat)) * 1.05))
    ax.set_yticks(np.arange(len(best)))
    ax.set_yticklabels([DISPLAY[m].replace(" + RetinaNet", "") for m in best["method"]])
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white" if mat[i, j] > 0.13 else "black", fontsize=8)
    ax.set_title("Best validation metrics by algorithm", loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("COCO score")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"figure_aitod_cvpr_metric_matrix.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    df = build_dataframe()
    df.to_csv(OUT / "aitod_cvpr_all_metrics.csv", index=False)
    (OUT / "aitod_cvpr_all_metrics.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    plot_trajectory(df)
    plot_gap(df)
    plot_decision_panel(df)


if __name__ == "__main__":
    main()
