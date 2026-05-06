#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a simple grouped comparison from exported run CSV.")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", default="visdrone")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("dataset", "").lower() != args.dataset.lower():
                continue
            if not row.get("model"):
                continue
            rows.append(row)

    labels = [row["model"] for row in rows]
    metrics = {
        "bbox_mAP": [float(row["bbox_mAP"]) for row in rows],
        "bbox_mAP_50": [float(row["bbox_mAP_50"]) for row in rows],
        "bbox_mAP_s": [float(row["bbox_mAP_s"]) for row in rows],
    }

    x = range(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width for i in x], metrics["bbox_mAP"], width=width, label="AP", color="#1f77b4")
    ax.bar(list(x), metrics["bbox_mAP_50"], width=width, label="AP50", color="#2ca02c")
    ax.bar([i + width for i in x], metrics["bbox_mAP_s"], width=width, label="APs", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Metric")
    ax.set_title(f"{args.dataset} Main Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    print(json.dumps({"output": str(args.output), "dataset": args.dataset, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
