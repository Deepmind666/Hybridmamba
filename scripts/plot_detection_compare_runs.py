#!/usr/bin/env python3
"""High-quality publication figure for two validation runs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker


def _apply_journal_rc(*, width_in: float = 6.8) -> None:
    """Apply journal-like style tuned for smooth, continuous curves."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "0.85",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            "lines.linewidth": 2.1,
            "lines.markersize": 0.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.figsize": (width_in, 3.6),
            "axes.prop_cycle": mpl.cycler(
                "color",
                ["#0B5FA5", "#DC6E2F", "#2D9D78", "#7E6FDC"],
            ),
        }
    )


def load_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "mAP": float(row["mAP"]),
                    "mAP50": float(row["mAP50"]),
                    "mAP75": float(row["mAP75"]),
                }
            )
    rows.sort(key=lambda r: r["epoch"])
    return rows


def _best_map(rows: list[dict[str, float]]) -> tuple[int, float]:
    i = max(range(len(rows)), key=lambda j: rows[j]["mAP"])
    return rows[i]["epoch"], rows[i]["mAP"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two val_progress.csv runs with publication style."
    )
    p.add_argument("--csv-a", type=Path, required=True, help="First run CSV.")
    p.add_argument(
        "--name-a",
        type=str,
        required=True,
        help='Panel legend name, e.g. "HybridMambaDet-B + RetinaNet".',
    )
    p.add_argument("--csv-b", type=Path, required=True)
    p.add_argument(
        "--name-b",
        type=str,
        required=True,
        help='Second method, e.g. "TinyViM-B + RetinaNet".',
    )
    p.add_argument("--out-png", type=Path, required=True)
    p.add_argument(
        "--dataset-line",
        type=str,
        default="VisDrone-DET · validation",
        help="Small figure note (dataset / split), not a long caption.",
    )
    p.add_argument(
        "--out-caption",
        type=Path,
        default=None,
        help="Optional .txt with a suggested LaTeX-ready caption (edit in manuscript).",
    )
    p.add_argument(
        "--width-in",
        type=float,
        default=6.8,
        help="Figure width in inches (double-column default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a = load_csv(args.csv_a.resolve())
    b = load_csv(args.csv_b.resolve())
    out = args.out_png.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    _apply_journal_rc(width_in=args.width_in)

    colors = ["#0072B2", "#D55E00"]
    c_a, c_b = colors[0], colors[1]

    ep_a, m_a = _best_map(a)
    ep_b, m_b = _best_map(b)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(args.width_in, 3.35), gridspec_kw={"wspace": 0.22})

    ax0.plot(
        [r["epoch"] for r in a],
        [r["mAP"] for r in a],
        color=c_a,
        linestyle="-",
        solid_capstyle="round",
        label=args.name_a,
    )
    ax0.plot(
        [r["epoch"] for r in b],
        [r["mAP"] for r in b],
        color=c_b,
        linestyle="-",
        solid_capstyle="round",
        label=args.name_b,
    )
    ax0.scatter([ep_a], [m_a], color=c_a, s=34, zorder=6, edgecolors="white", linewidths=0.9)
    ax0.scatter([ep_b], [m_b], color=c_b, s=34, zorder=6, edgecolors="white", linewidths=0.9)

    ax0.set_ylabel(r"mAP @ IoU $[0.50,\,0.95]$")
    ax0.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], integer=False))
    ax0.grid(True, axis="both", linestyle="--")
    ax0.text(
        0.02,
        0.96,
        "(a)",
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontweight="medium",
    )
    leg0 = ax0.legend(loc="lower right", handlelength=2.6)
    leg0.get_frame().set_linewidth(0.5)

    ax1.plot(
        [r["epoch"] for r in a],
        [r["mAP50"] for r in a],
        color=c_a,
        linestyle="--",
        dashes=(5, 3),
        solid_capstyle="round",
        label=args.name_a,
    )
    ax1.plot(
        [r["epoch"] for r in b],
        [r["mAP50"] for r in b],
        color=c_b,
        linestyle="--",
        dashes=(5, 3),
        solid_capstyle="round",
        label=args.name_b,
    )
    ax1.set_xlabel("Training epoch")
    ax1.set_ylabel(r"AP@0.50")
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], integer=False))
    ax1.grid(True, axis="both", linestyle="--")
    ax1.text(
        0.02,
        0.96,
        "(b)",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontweight="medium",
    )
    leg1 = ax1.legend(loc="lower right", handlelength=2.6)
    leg1.get_frame().set_linewidth(0.5)

    fig.text(0.5, 0.01, args.dataset_line, ha="center", va="bottom", fontsize=9, color="0.25")
    plt.subplots_adjust(bottom=0.18, top=0.97, left=0.08, right=0.99)

    fig.savefig(out, format="png")
    plt.close(fig)
    print(f"Wrote {out}")

    if args.out_caption:
        cap_path = args.out_caption.resolve()
        cap_path.parent.mkdir(parents=True, exist_ok=True)
        # Neutral wording for manuscript; replace \METHOD with your macro if needed.
        text = (
            "\\textbf{Validation metrics on VisDrone-DET.} "
            "(a) COCO-style mean average precision (mAP) averaged over IoU thresholds from 0.50 to 0.95. "
            "(b) Average precision at IoU$=$0.50. "
            "Both systems use RetinaNet with FPN and the same evaluation protocol on the validation split. "
            f"The highlighted points indicate the peak mAP for each model "
            f"({args.name_a}: {m_a:.3f} at epoch {ep_a}; "
            f"{args.name_b}: {m_b:.3f} at epoch {ep_b}). "
            "Training schedules and data preprocessing are specified in the configuration files. "
            "Edit this sentence in the final manuscript if the two runs use different input resolutions or schedules."
        )
        cap_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote suggested caption: {cap_path}")


if __name__ == "__main__":
    main()
