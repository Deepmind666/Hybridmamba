#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from config_utils import load_python_config, resolve_runtime_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch preflight all Hybrid Mamba detection configs.")
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("C:/mamba/code/tinyvim/detection/configs"),
        help="Directory containing experiment configs.",
    )
    return parser.parse_args()


def inspect_config(config_path: Path) -> Dict[str, object]:
    cfg = load_python_config(config_path)
    model_cfg = cfg.get("model", {})
    backbone_cfg = model_cfg.get("backbone", {})
    schema_issues = []
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
        if key not in cfg:
            schema_issues.append(f"missing:{key}")
    for key in forbidden_keys:
        if key in cfg:
            schema_issues.append(f"forbidden:{key}")

    dataset_checks: List[Dict[str, object]] = []
    for split in ("train", "val", "test"):
        split_cfg = cfg.get("data", {}).get(split, {})
        for key in ("ann_file", "img_prefix"):
            raw_path = split_cfg.get(key)
            if raw_path:
                resolved = resolve_runtime_path(config_path, raw_path)
                dataset_checks.append(
                    {
                        "split": split,
                        "kind": key,
                        "path": str(resolved),
                        "exists": resolved.exists(),
                    }
                )

    checkpoint = backbone_cfg.get("init_cfg", {}).get("checkpoint", "")
    checkpoint_path = resolve_runtime_path(config_path, checkpoint) if checkpoint else None

    return {
        "config": str(config_path.resolve()),
        "backbone": backbone_cfg.get("type", ""),
        "checkpoint": {
            "path": str(checkpoint_path) if checkpoint_path else "",
            "exists": checkpoint_path.exists() if checkpoint_path else False,
        },
        "schema_issues": schema_issues,
        "dataset_checks": dataset_checks,
    }


def main() -> None:
    args = parse_args()
    configs = sorted(args.config_root.glob("retinanet_*.py"))
    results = [inspect_config(path) for path in configs]
    summary = {
        "config_count": len(results),
        "configs": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
