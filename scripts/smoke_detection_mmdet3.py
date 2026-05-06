#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from mmengine.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short single-GPU MMDet3 smoke pass.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def detection_root(config_path: Path) -> Path:
    return config_path.resolve().parents[1]


def ensure_import_path(config_path: Path) -> None:
    root = detection_root(config_path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    import model  # noqa: F401


def make_manifest(args: argparse.Namespace, cfg: Config) -> dict:
    run_id = f"smoke3_{args.config.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    backbone_cfg = cfg.model.get("backbone", {})
    init_cfg = backbone_cfg.get("init_cfg", {})
    dataset_cfg = cfg.train_dataloader.get("dataset", {})
    ann_file = dataset_cfg.get("ann_file", "")
    dataset_name = Path(str(ann_file)).parts[-2] if ann_file else ""
    return {
        "run_id": run_id,
        "config": str(args.config.resolve()),
        "work_dir": str(args.work_dir.resolve()),
        "dataset": dataset_name,
        "dataset_config": args.config.stem,
        "model": backbone_cfg.get("type", ""),
        "detector": cfg.model.get("type", ""),
        "stack": "mmdet3-cu128-blackwell",
        "backbone_type": backbone_cfg.get("type", ""),
        "backbone_init_checkpoint": init_cfg.get("checkpoint", ""),
        "max_iters": args.max_iters,
        "seed": args.seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    }


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.work_dir = args.work_dir.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MMDet3 smoke test.")
    try:
        _ = torch.randn(1, device="cuda")
    except Exception as exc:
        raise RuntimeError(
            "CUDA is visible but unusable for the current PyTorch build. "
            "Use scripts/check_runtime_stack.py to inspect the active env."
        ) from exc

    ensure_import_path(args.config)
    from mmdet.utils import register_all_modules
    from mmengine.logging import MMLogger
    from mmengine.optim import build_optim_wrapper
    from mmengine.runner import Runner
    from mmdet.registry import MODELS

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(args.config))
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.train_dataloader.pin_memory = False
    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    cfg.val_dataloader.pin_memory = False
    cfg.randomness = dict(seed=args.seed)
    if hasattr(cfg, "env_cfg"):
        cfg.env_cfg.setdefault("mp_cfg", {})["mp_start_method"] = "fork"
        cfg.env_cfg["cudnn_benchmark"] = False

    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(args, cfg)
    (args.work_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger = MMLogger.get_instance("smoke3", log_file=str(args.work_dir / "train.log"), log_level="INFO")

    train_loader = Runner.build_dataloader(cfg.train_dataloader, seed=args.seed)
    model = MODELS.build(cfg.model)
    if hasattr(model, "init_weights"):
        model.init_weights()
    model = model.cuda()
    optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)

    model.train()
    timings = []
    losses = []
    data_iter = iter(train_loader)
    for step in range(1, args.max_iters + 1):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data = next(data_iter)
        start = time.perf_counter()
        with optim_wrapper.optim_context(model):
            data = model.data_preprocessor(data, training=True)
            losses_dict = model(**data, mode="loss")
            parsed_loss, _ = model.parse_losses(losses_dict)
        optim_wrapper.update_params(parsed_loss)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        losses.append(float(parsed_loss.detach().cpu()))
        logger.info("iter=%d loss=%.6f time=%.4fs", step, losses[-1], elapsed)

    metrics = {
        "status": "ok",
        "iters": len(losses),
        "mean_loss": mean(losses),
        "last_loss": losses[-1],
        "mean_iter_time_sec": mean(timings),
        "cuda_device": torch.cuda.get_device_name(0),
    }
    (args.work_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
