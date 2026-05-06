#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MMDet3 train and validation curves from scalars.json.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def find_scalars(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("scalars.json"))
    if not candidates:
        raise FileNotFoundError(f"No scalars.json found under {run_dir}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    scalars_path = find_scalars(args.run_dir.resolve())
    train_steps, train_loss = [], []
    val_steps = []
    val_metrics = defaultdict(list)

    with scalars_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if "loss" in record and "coco/bbox_mAP" not in record:
                train_steps.append(record.get("step", len(train_steps) + 1))
                train_loss.append(record["loss"])
            if "coco/bbox_mAP" in record:
                step = record.get("step", len(val_steps) + 1)
                val_steps.append(step)
                for key in ("coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_s"):
                    val_metrics[key].append(record.get(key, 0.0))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].plot(train_steps, train_loss, label="train_loss", color="#1f77b4")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    for key, color in zip(("coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_s"), ("#2ca02c", "#d62728", "#9467bd")):
        if val_metrics[key]:
            axes[1].plot(val_steps, val_metrics[key], marker="o", label=key, color=color)
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch Step")
    axes[1].set_ylabel("Metric")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    print(json.dumps({"output": str(args.output), "source": str(scalars_path)}, indent=2))


if __name__ == "__main__":
    main()

