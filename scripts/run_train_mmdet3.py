#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner

_original_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_compat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch MMDet3 training for the Blackwell migration path.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-mem-gb", type=float, default=float(os.environ.get("GPU_MEM_GB", "30")))
    parser.add_argument("--torch-num-threads", type=int, default=int(os.environ.get("TORCH_NUM_THREADS", "8")))
    parser.add_argument("--torch-num-interop-threads", type=int, default=int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "2")))
    return parser.parse_args()


def detection_root(config_path: Path) -> Path:
    return config_path.resolve().parents[1]


def ensure_import_path(config_path: Path) -> None:
    root = detection_root(config_path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    import model  # noqa: F401


def _clamp_loader(loader_cfg) -> None:
    if loader_cfg is None:
        return
    loader_cfg["num_workers"] = 0
    loader_cfg["persistent_workers"] = False
    loader_cfg["pin_memory"] = False


def apply_local_safe_mode(cfg: Config, args: argparse.Namespace) -> list[str]:
    changes: list[str] = []

    for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if hasattr(cfg, loader_name):
            _clamp_loader(getattr(cfg, loader_name))
            changes.append(f"{loader_name}: workers=0 persistent_workers=False pin_memory=False")

    if hasattr(cfg, "env_cfg"):
        cfg.env_cfg["cudnn_benchmark"] = False
        cfg.env_cfg.setdefault("mp_cfg", {})["opencv_num_threads"] = 0
        changes.append("env_cfg: cudnn_benchmark=False opencv_num_threads=0")

    if args.torch_num_threads > 4:
        args.torch_num_threads = 4
        changes.append("torch_num_threads clamped to 4")
    if args.torch_num_interop_threads > 1:
        args.torch_num_interop_threads = 1
        changes.append("torch_num_interop_threads clamped to 1")
    if args.gpu_mem_gb > 24:
        args.gpu_mem_gb = 24
        changes.append("gpu_mem_gb clamped to 24")

    return changes


def runner_warmup(runner: Runner, cfg: Config, seed: int, warmup_iters: int) -> None:
    if warmup_iters <= 0:
        return

    train_loader = Runner.build_dataloader(cfg.train_dataloader, seed=seed)
    data_iter = iter(train_loader)
    model = runner.model
    optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
    model.train()

    print(f"Runner warm-up active: {warmup_iters} iter(s)")
    for step in range(1, warmup_iters + 1):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data = next(data_iter)

        with optim_wrapper.optim_context(model):
            data = model.data_preprocessor(data, training=True)
            losses_dict = model(**data, mode="loss")
            parsed_loss, _ = model.parse_losses(losses_dict)
        optim_wrapper.update_params(parsed_loss)
        torch.cuda.synchronize()
        print(f"Warm-up iter {step}: loss={float(parsed_loss.detach().cpu()):.6f}")

    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    local_safe_mode = os.environ.get("MAMBA_LOCAL_SAFE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    runner_warmup_iters = int(os.environ.get("MAMBA_RUNNER_WARMUP_ITERS", "0") or "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads and args.torch_num_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_num_interop_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MMDet3 training.")
    try:
        _ = torch.randn(1, device="cuda")
    except Exception as exc:
        raise RuntimeError("CUDA is visible but unusable for the current PyTorch build.") from exc

    if args.gpu_mem_gb and args.gpu_mem_gb > 0:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        requested_bytes = int(args.gpu_mem_gb * (1024 ** 3))
        if requested_bytes < total_bytes:
            fraction = requested_bytes / total_bytes
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)

    ensure_import_path(args.config)
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(args.config))
    if args.work_dir is not None:
        cfg.work_dir = str(args.work_dir.resolve())
    cfg.randomness = dict(seed=args.seed)
    if hasattr(cfg, "env_cfg"):
        cfg.env_cfg.setdefault("mp_cfg", {})["mp_start_method"] = "fork"
    safe_mode_changes: list[str] = []
    if local_safe_mode:
        safe_mode_changes = apply_local_safe_mode(cfg, args)

    runner = Runner.from_cfg(cfg)
    if local_safe_mode and safe_mode_changes:
        print("Local safe mode active:")
        for change in safe_mode_changes:
            print(f"- {change}")
    runner_warmup(runner, cfg, args.seed, runner_warmup_iters)
    runner.train()


if __name__ == "__main__":
    main()
