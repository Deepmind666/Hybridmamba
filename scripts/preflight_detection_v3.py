#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from config_utils import load_python_config, resolve_runtime_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static preflight for MMDet3 HybridMamba configs.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def collect_dataset_checks(config_path: Path, cfg: Dict[str, object]) -> List[Dict[str, object]]:
    checks: List[Dict[str, object]] = []
    for loader_key, split in (("train_dataloader", "train"), ("val_dataloader", "val"), ("test_dataloader", "test")):
        loader_cfg = cfg.get(loader_key, {})
        dataset_cfg = loader_cfg.get("dataset", {})
        ann_file = dataset_cfg.get("ann_file")
        data_prefix = dataset_cfg.get("data_prefix", {})
        img_prefix = data_prefix.get("img") if isinstance(data_prefix, dict) else None
        data_root = dataset_cfg.get("data_root", "")
        if ann_file:
            ann_path = resolve_runtime_path(config_path, str(Path(data_root) / ann_file))
            checks.append({"type": "ann_file", "split": split, "path": str(ann_path), "exists": ann_path.exists()})
        if img_prefix:
            img_prefix_path = Path(str(img_prefix))
            if img_prefix_path.is_absolute() or str(img_prefix).startswith("."):
                img_path = resolve_runtime_path(config_path, str(img_prefix))
            else:
                img_path = resolve_runtime_path(config_path, str(Path(data_root) / img_prefix))
            checks.append({"type": "img_prefix", "split": split, "path": str(img_path), "exists": img_path.exists()})
    return checks


def collect_schema_checks(cfg: Dict[str, object]) -> List[Dict[str, object]]:
    checks: List[Dict[str, object]] = []
    required_keys = ("model", "optim_wrapper", "train_dataloader", "val_dataloader", "test_dataloader", "train_cfg")
    forbidden_keys = ("data", "optimizer", "optimizer_config", "runner")
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
