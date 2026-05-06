#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


KEY_MAP = {
    "coco/bbox_mAP": "bbox_mAP",
    "coco/bbox_mAP_50": "bbox_mAP_50",
    "coco/bbox_mAP_75": "bbox_mAP_75",
    "coco/bbox_mAP_s": "bbox_mAP_s",
    "coco/bbox_mAP_m": "bbox_mAP_m",
    "coco/bbox_mAP_l": "bbox_mAP_l",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the latest MMDet3 validation metrics from scalars.json.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def find_scalars_json(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("scalars.json"))
    if not candidates:
        raise FileNotFoundError(f"No scalars.json found under {run_dir}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    scalars_path = find_scalars_json(run_dir)

    latest = {}
    with scalars_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            if "coco/bbox_mAP" not in record:
                continue
            latest = record

    if not latest:
        raise RuntimeError(f"No validation metrics found in {scalars_path}")

    metrics = {"status": "ok", "source": str(scalars_path), "step": latest.get("step", -1)}
    for source_key, target_key in KEY_MAP.items():
        metrics[target_key] = latest.get(source_key, "")

    output_json = args.output_json or (run_dir / "eval_metrics.json")
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
