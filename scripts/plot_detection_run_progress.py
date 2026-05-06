#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


VAL_PATTERN = re.compile(
    r"Epoch\(val\)\s+\[(?P<epoch>\d+)\]\[\d+/\d+\]\s+"
    r"coco/bbox_mAP:\s+(?P<map>[0-9.]+)\s+"
    r"coco/bbox_mAP_50:\s+(?P<map50>[0-9.]+)\s+"
    r"coco/bbox_mAP_75:\s+(?P<map75>[0-9.]+)\s+"
    r"coco/bbox_mAP_s:\s+(?P<maps>[0-9.]+)\s+"
    r"coco/bbox_mAP_m:\s+(?P<mapm>[0-9.]+)\s+"
    r"coco/bbox_mAP_l:\s+(?P<mapl>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RetinaNet/TinyViM validation progress from train.log."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory that contains train.log",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for csv and figures (default: <run-dir>/plots)",
    )
    return parser.parse_args()


def parse_train_log(log_path: Path) -> list[dict[str, float]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    rows: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = VAL_PATTERN.search(line)
            if not match:
                continue
            rows.append(
                {
                    "epoch": int(match.group("epoch")),
                    "mAP": float(match.group("map")),
                    "mAP50": float(match.group("map50")),
                    "mAP75": float(match.group("map75")),
                    "mAP_s": float(match.group("maps")),
                    "mAP_m": float(match.group("mapm")),
                    "mAP_l": float(match.group("mapl")),
                }
            )
    return rows


def dump_csv(rows: list[dict[str, float]], csv_path: Path) -> None:
    fieldnames = ["epoch", "mAP", "mAP50", "mAP75", "mAP_s", "mAP_m", "mAP_l"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_main_curve(rows: list[dict[str, float]], png_path: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    map_vals = [row["mAP"] for row in rows]
    map50_vals = [row["mAP50"] for row in rows]
    map75_vals = [row["mAP75"] for row in rows]

    best_idx = max(range(len(rows)), key=lambda i: map_vals[i])
    best_epoch = epochs[best_idx]
    best_map = map_vals[best_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, map_vals, marker="o", label="mAP")
    plt.plot(epochs, map50_vals, marker="o", label="mAP50")
    plt.plot(epochs, map75_vals, marker="o", label="mAP75")
    plt.scatter([best_epoch], [best_map], color="red", zorder=5, label=f"best mAP={best_map:.3f} @ep{best_epoch}")
    plt.title("Validation AP Progress")
    plt.xlabel("Epoch")
    plt.ylabel("AP")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


def plot_scale_curve(rows: list[dict[str, float]], png_path: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    maps_vals = [row["mAP_s"] for row in rows]
    mapm_vals = [row["mAP_m"] for row in rows]
    mapl_vals = [row["mAP_l"] for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, maps_vals, marker="o", label="mAP_s")
    plt.plot(epochs, mapm_vals, marker="o", label="mAP_m")
    plt.plot(epochs, mapl_vals, marker="o", label="mAP_l")
    plt.title("Validation AP by Scale")
    plt.xlabel("Epoch")
    plt.ylabel("AP")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log"
    rows = parse_train_log(log_path)
    if not rows:
        raise RuntimeError(f"No validation lines matched in {log_path}")

    rows.sort(key=lambda x: x["epoch"])
    first_ep = rows[0]["epoch"]
    if first_ep > 1:
        print(
            f"Note: first validation line in this log is epoch {first_ep}. "
            "Earlier epochs are usually still in the full train.log on the training host "
            "(partial copy, tail, or late attach); fetch that file if you need epoch 1 onward.",
            file=sys.stderr,
        )

    csv_path = out_dir / "val_progress.csv"
    main_png = out_dir / "val_ap_progress.png"
    scale_png = out_dir / "val_ap_scale_progress.png"

    dump_csv(rows, csv_path)
    plot_main_curve(rows, main_png)
    plot_scale_curve(rows, scale_png)

    best_row = max(rows, key=lambda r: r["mAP"])
    print(f"Parsed validation points: {len(rows)}")
    print(f"Best mAP: {best_row['mAP']:.3f} @ epoch {best_row['epoch']}")
    print(f"CSV: {csv_path}")
    print(f"Figure1: {main_png}")
    print(f"Figure2: {scale_png}")


if __name__ == "__main__":
    main()

