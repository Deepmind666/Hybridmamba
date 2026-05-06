#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config_utils import load_python_config, resolve_runtime_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a run manifest from an MMDet3 config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", default="", help="Optional dataset label override.")
    parser.add_argument("--run-id", default="", help="Optional run id override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_python_config(args.config)
    model_cfg = cfg.get("model", {})
    backbone_cfg = model_cfg.get("backbone", {})
    run_id = args.run_id or f"{args.config.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint = backbone_cfg.get("init_cfg", {}).get("checkpoint", "")
    checkpoint_path = str(resolve_runtime_path(args.config, checkpoint)) if checkpoint else ""

    dataset_name = args.dataset
    if not dataset_name:
        ann_file = cfg.get("train_dataloader", {}).get("dataset", {}).get("ann_file", "")
        dataset_name = Path(ann_file).parts[-2] if ann_file else ""

    manifest = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(args.config.resolve()),
        "dataset": dataset_name,
        "model": backbone_cfg.get("type", ""),
        "detector": model_cfg.get("type", ""),
        "checkpoint": checkpoint_path,
        "work_dir": cfg.get("work_dir", ""),
        "stack": "mmdet3-cu128-blackwell",
        "expected_eval_file": "eval_metrics.json",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
