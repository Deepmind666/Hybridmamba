#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from config_utils import load_python_config, resolve_runtime_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static preflight for TinyViM detection configs.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def collect_dataset_checks(config_path: Path, cfg: Dict[str, object]) -> List[Dict[str, object]]:
    checks: List[Dict[str, object]] = []
    data_cfg = cfg.get("data", {})
    for split in ("train", "val", "test"):
        split_cfg = data_cfg.get(split, {})
        ann_file = split_cfg.get("ann_file")
        img_prefix = split_cfg.get("img_prefix")
        if ann_file:
            ann_path = resolve_runtime_path(config_path, ann_file)
            checks.append({"type": "ann_file", "split": split, "path": str(ann_path), "exists": ann_path.exists()})
        if img_prefix:
            img_path = resolve_runtime_path(config_path, img_prefix)
            checks.append({"type": "img_prefix", "split": split, "path": str(img_path), "exists": img_path.exists()})
    return checks


def collect_schema_checks(cfg: Dict[str, object]) -> List[Dict[str, object]]:
    checks: List[Dict[str, object]] = []
    required_keys = ("data", "optimizer", "optimizer_config", "runner")
    forbidden_keys = (
        "optim_wrapper",
        "train_dataloader",
        "val_dataloader",
        "test_dataloader",
        "default_scope",
        "default_hooks",
        "env_cfg",
        "vis_backends",
        "visualizer",
        "log_processor",
        "resume",
        "param_scheduler",
    )
    for key in required_keys:
        checks.append({"type": "required_key", "key": key, "present": key in cfg})
    for key in forbidden_keys:
        checks.append({"type": "forbidden_key", "key": key, "present": key in cfg})
    return checks


def main() -> None:
    args = parse_args()
    cfg = load_python_config(args.config)
    model_cfg = cfg.get("model", {})
    backbone_cfg = model_cfg.get("backbone", {})
    checkpoint = backbone_cfg.get("init_cfg", {}).get("checkpoint")
    checkpoint_check = None
    if checkpoint:
        checkpoint_path = resolve_runtime_path(args.config, checkpoint)
        checkpoint_check = {"path": str(checkpoint_path), "exists": checkpoint_path.exists()}

    payload = {
        "config": str(args.config.resolve()),
        "model_type": model_cfg.get("type"),
        "backbone_type": backbone_cfg.get("type"),
        "work_dir": cfg.get("work_dir", ""),
        "checkpoint": checkpoint_check,
        "schema_checks": collect_schema_checks(cfg),
        "dataset_checks": collect_dataset_checks(args.config, cfg),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
