#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.logging import MMLogger
from mmengine.logging import MessageHub
from mmengine.optim import build_optim_wrapper
from mmengine.registry import PARAM_SCHEDULERS
from mmengine.runner import Runner
from mmengine.runner.checkpoint import weights_to_cpu
from mmengine.utils import apply_to

_original_torch_load = torch.load
LOCAL_SAFE_GPU_MEM_GB = 28.0


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_compat


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch MMDet3 training with a manual loop that avoids Runner.train().")
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-mem-gb",
                        type=float,
                        default=float(os.environ.get("GPU_MEM_GB", str(LOCAL_SAFE_GPU_MEM_GB))))
    parser.add_argument("--torch-num-threads",
                        type=int,
                        default=int(os.environ.get("TORCH_NUM_THREADS", "8")))
    parser.add_argument("--torch-num-interop-threads",
                        type=int,
                        default=int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "2")))
    parser.add_argument("--adaptive-guard",
                        action="store_true",
                        default=env_flag("MAMBA_ADAPTIVE_GUARD", False))
    parser.add_argument("--guard-gpu-util-pct",
                        type=float,
                        default=env_float("MAMBA_GUARD_GPU_UTIL_PCT", 80.0))
    parser.add_argument("--guard-cpu-util-pct",
                        type=float,
                        default=env_float("MAMBA_GUARD_CPU_UTIL_PCT", 80.0))
    parser.add_argument("--guard-resume-util-pct",
                        type=float,
                        default=env_float("MAMBA_GUARD_RESUME_UTIL_PCT", 70.0))
    parser.add_argument("--guard-temp-c",
                        type=float,
                        default=env_float("MAMBA_GUARD_TEMP_C", 78.0))
    parser.add_argument("--guard-memory-pct",
                        type=float,
                        default=env_float("MAMBA_GUARD_MEMORY_PCT", 92.0))
    parser.add_argument("--guard-check-interval-sec",
                        type=float,
                        default=env_float("MAMBA_GUARD_CHECK_INTERVAL_SEC", 2.0))
    parser.add_argument("--guard-cooldown-sec",
                        type=float,
                        default=env_float("MAMBA_GUARD_COOLDOWN_SEC", 20.0))
    parser.add_argument("--guard-log-interval-sec",
                        type=float,
                        default=env_float("MAMBA_GUARD_LOG_INTERVAL_SEC", 120.0))
    return parser.parse_args()


def detection_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if (parent / "model.py").exists() or (parent / "model").exists():
            return parent
        if (parent / "backbones").exists():
            return parent
    return resolved.parents[1]


def ensure_import_path(config_path: Path) -> None:
    root = detection_root(config_path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    imported = False
    if (root / "model.py").exists() or (root / "model").exists():
        import model  # noqa: F401
        imported = True
    if (root / "backbones").exists():
        import backbones  # noqa: F401
        imported = True
    if not imported:
        raise RuntimeError(f"No MMDet model registry module found under {root}")


def _clamp_loader(loader_cfg) -> None:
    if loader_cfg is None:
        return
    loader_cfg["num_workers"] = 0
    loader_cfg["persistent_workers"] = False
    loader_cfg["pin_memory"] = False


def apply_local_safe_runtime_args(args: argparse.Namespace) -> list[str]:
    changes: list[str] = []

    if args.torch_num_threads > 4:
        args.torch_num_threads = 4
        changes.append("torch_num_threads clamped to 4")
    if args.torch_num_interop_threads > 1:
        args.torch_num_interop_threads = 1
        changes.append("torch_num_interop_threads clamped to 1")
    if args.gpu_mem_gb > LOCAL_SAFE_GPU_MEM_GB:
        args.gpu_mem_gb = LOCAL_SAFE_GPU_MEM_GB
        changes.append(f"gpu_mem_gb clamped to {LOCAL_SAFE_GPU_MEM_GB:.0f}")

    return changes


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

    return changes


def flatten_schedulers(param_schedulers: Any) -> list[Any]:
    if param_schedulers is None:
        return []
    if isinstance(param_schedulers, dict):
        flat: list[Any] = []
        for scheds in param_schedulers.values():
            flat.extend(flatten_schedulers(scheds))
        return flat
    if isinstance(param_schedulers, (list, tuple)):
        flat: list[Any] = []
        for scheduler in param_schedulers:
            flat.extend(flatten_schedulers(scheduler))
        return flat
    return [param_schedulers]


def step_schedulers(param_schedulers: Any, *, by_epoch: bool) -> None:
    for scheduler in flatten_schedulers(param_schedulers):
        if getattr(scheduler, "by_epoch", True) == by_epoch:
            scheduler.step()


def first_optimizer_lr(optim_wrapper) -> float:
    optimizer = optim_wrapper.optimizer
    return float(optimizer.param_groups[0]["lr"])


def get_scalar_float(message_hub, key: str) -> float | None:
    try:
        return float(message_hub.get_scalar(key).current())
    except Exception:
        return None


def make_manifest(args: argparse.Namespace, cfg: Config) -> dict:
    backbone_cfg = cfg.model.get("backbone", {})
    init_cfg = backbone_cfg.get("init_cfg", {})
    dataset_cfg = cfg.train_dataloader.get("dataset", {})
    ann_file = dataset_cfg.get("ann_file", "")
    dataset_name = Path(str(ann_file)).parts[-2] if ann_file else ""
    max_epochs = int(cfg.train_cfg.get("max_epochs", 0))
    return {
        "run_id": args.work_dir.name,
        "config": str(args.config.resolve()),
        "work_dir": str(args.work_dir.resolve()),
        "dataset": dataset_name,
        "dataset_config": args.config.stem,
        "model": backbone_cfg.get("type", ""),
        "detector": cfg.model.get("type", ""),
        "stack": "mmdet3-cu128-blackwell-manual",
        "backbone_type": backbone_cfg.get("type", ""),
        "backbone_init_checkpoint": init_cfg.get("checkpoint", ""),
        "max_epochs": max_epochs,
        "seed": args.seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "manual_train_loop": True,
        "adaptive_guard": {
            "enabled": bool(args.adaptive_guard),
            "gpu_util_pct": args.guard_gpu_util_pct,
            "cpu_util_pct": args.guard_cpu_util_pct,
            "resume_util_pct": args.guard_resume_util_pct,
            "temp_c": args.guard_temp_c,
            "memory_pct": args.guard_memory_pct,
            "check_interval_sec": args.guard_check_interval_sec,
            "cooldown_sec": args.guard_cooldown_sec,
        },
    }


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days} days, {hours}:{minutes:02d}:{secs:02d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"


def emit_log(log_path: Path, message: str) -> None:
    line = f"{time.strftime('%m/%d %H:%M:%S')} - mmengine - INFO - {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def query_gpu_stats() -> dict[str, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        return None
    try:
        util, mem_used, mem_total, temp, power = [float(part.strip()) for part in line.split(",")[:5]]
    except Exception:
        return None
    memory_pct = 100.0 * mem_used / max(mem_total, 1.0)
    return {
        "gpu_util_pct": util,
        "gpu_memory_used_mb": mem_used,
        "gpu_memory_total_mb": mem_total,
        "gpu_memory_pct": memory_pct,
        "gpu_temp_c": temp,
        "gpu_power_w": power,
    }


def query_cpu_util_pct() -> float:
    try:
        load_1min = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        return min(100.0, 100.0 * load_1min / cpu_count)
    except Exception:
        return 0.0


def guard_violation(stats: dict[str, float], args: argparse.Namespace, *, resume: bool = False) -> list[str]:
    util_limit = args.guard_resume_util_pct if resume else args.guard_gpu_util_pct
    cpu_limit = args.guard_resume_util_pct if resume else args.guard_cpu_util_pct
    temp_limit = max(args.guard_temp_c - 3.0, 0.0) if resume else args.guard_temp_c
    violations: list[str] = []
    if stats.get("gpu_util_pct", 0.0) > util_limit:
        violations.append(f"gpu_util={stats['gpu_util_pct']:.1f}%>{util_limit:.1f}%")
    if stats.get("cpu_util_pct", 0.0) > cpu_limit:
        violations.append(f"cpu_util={stats['cpu_util_pct']:.1f}%>{cpu_limit:.1f}%")
    if stats.get("gpu_temp_c", 0.0) > temp_limit:
        violations.append(f"gpu_temp={stats['gpu_temp_c']:.1f}C>{temp_limit:.1f}C")
    if stats.get("gpu_memory_pct", 0.0) > args.guard_memory_pct:
        violations.append(f"gpu_mem={stats['gpu_memory_pct']:.1f}%>{args.guard_memory_pct:.1f}%")
    return violations


def collect_guard_stats() -> dict[str, float]:
    stats = query_gpu_stats() or {}
    stats["cpu_util_pct"] = query_cpu_util_pct()
    return stats


def format_guard_stats(stats: dict[str, float]) -> str:
    return (
        f"gpu_util={stats.get('gpu_util_pct', 0.0):.1f}% "
        f"cpu_util={stats.get('cpu_util_pct', 0.0):.1f}% "
        f"gpu_temp={stats.get('gpu_temp_c', 0.0):.1f}C "
        f"gpu_mem={stats.get('gpu_memory_used_mb', 0.0):.0f}/"
        f"{stats.get('gpu_memory_total_mb', 0.0):.0f}MiB "
        f"power={stats.get('gpu_power_w', 0.0):.1f}W"
    )


def adaptive_guard_pause(args: argparse.Namespace, log_path: Path, state: dict[str, float]) -> None:
    if not args.adaptive_guard:
        return
    now = time.monotonic()
    if now - state.get("last_check", 0.0) < args.guard_check_interval_sec:
        return
    state["last_check"] = now
    stats = collect_guard_stats()
    violations = guard_violation(stats, args, resume=False)
    if not violations:
        if now - state.get("last_status_log", 0.0) >= args.guard_log_interval_sec:
            emit_log(log_path, f"Adaptive guard status: {format_guard_stats(stats)}")
            state["last_status_log"] = now
        return

    emit_log(log_path, "Adaptive guard cooldown: " + "; ".join(violations) + f"; {format_guard_stats(stats)}")
    cooldowns = 0
    while True:
        cooldowns += 1
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(max(args.guard_cooldown_sec, 1.0))
        stats = collect_guard_stats()
        resume_violations = guard_violation(stats, args, resume=True)
        if not resume_violations:
            emit_log(
                log_path,
                f"Adaptive guard resumed after {cooldowns} cooldown(s): {format_guard_stats(stats)}",
            )
            state["last_status_log"] = time.monotonic()
            state["last_check"] = time.monotonic()
            return
        emit_log(log_path, "Adaptive guard still cooling: " + "; ".join(resume_violations) + f"; {format_guard_stats(stats)}")


def build_evaluator(cfg: Config, dataloader) -> Evaluator:
    evaluator = Evaluator(cfg.val_evaluator)
    if hasattr(dataloader.dataset, "metainfo"):
        evaluator.dataset_meta = dataloader.dataset.metainfo
    return evaluator


def run_validation(model, dataloader, cfg: Config, epoch: int, log_path: Path) -> dict:
    evaluator = build_evaluator(cfg, dataloader)
    model.eval()
    total_iters = len(dataloader)
    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader, start=1):
            outputs = model.val_step(data_batch)
            evaluator.process(data_samples=outputs, data_batch=data_batch)
    metrics = evaluator.evaluate(len(dataloader.dataset))
    torch.cuda.empty_cache()
    emit_log(
        log_path,
        "Epoch(val)   "
        f"[{epoch}][{total_iters}/{total_iters}]  "
        f"coco/bbox_mAP: {float(metrics.get('coco/bbox_mAP', 0.0)):.3f}  "
        f"coco/bbox_mAP_50: {float(metrics.get('coco/bbox_mAP_50', 0.0)):.3f}  "
        f"coco/bbox_mAP_75: {float(metrics.get('coco/bbox_mAP_75', 0.0)):.3f}  "
        f"coco/bbox_mAP_s: {float(metrics.get('coco/bbox_mAP_s', 0.0)):.3f}  "
        f"coco/bbox_mAP_m: {float(metrics.get('coco/bbox_mAP_m', 0.0)):.3f}  "
        f"coco/bbox_mAP_l: {float(metrics.get('coco/bbox_mAP_l', 0.0)):.3f}")
    model.train()
    return metrics


def is_better(current: float, best: float | None, rule: str, min_delta: float) -> bool:
    if best is None:
        return True
    if rule == "greater":
        return current > best + min_delta
    return current < best - min_delta


def save_checkpoint(work_dir: Path,
                    model,
                    optim_wrapper,
                    param_schedulers,
                    message_hub: MessageHub,
                    epoch: int,
                    global_iter: int,
                    latest_metrics: dict | None,
                    filename: str) -> Path:
    meta = {"manual_loop": True}
    if latest_metrics is not None:
        meta["latest_metrics"] = latest_metrics
    meta.update(epoch=epoch, iter=global_iter, time=time.strftime("%Y%m%d_%H%M%S"))

    checkpoint = {
        "meta": meta,
        "state_dict": weights_to_cpu(model.state_dict()),
        "optimizer": apply_to(
            optim_wrapper.state_dict(),
            lambda x: hasattr(x, "cpu"),
            lambda x: x.cpu(),
        ),
        "message_hub": apply_to(
            message_hub.state_dict(),
            lambda x: hasattr(x, "cpu"),
            lambda x: x.cpu(),
        ),
        "param_schedulers": [scheduler.state_dict() for scheduler in flatten_schedulers(param_schedulers)],
    }
    checkpoint_path = work_dir / filename
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def copy_checkpoint_payload(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def prune_old_epoch_checkpoints(work_dir: Path, max_keep: int) -> None:
    if max_keep <= 0:
        return
    epoch_ckpts = sorted(
        [path for path in work_dir.glob("epoch_*.pth") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
    )
    while len(epoch_ckpts) > max_keep:
        oldest = epoch_ckpts.pop(0)
        oldest.unlink(missing_ok=True)


def build_param_schedulers(param_scheduler_cfg: Any, optim_wrapper, epoch_length: int) -> list[Any]:
    if param_scheduler_cfg is None:
        return []
    cfgs = param_scheduler_cfg if isinstance(param_scheduler_cfg, (list, tuple)) else [param_scheduler_cfg]
    schedulers = []
    for scheduler_cfg in cfgs:
        scheduler_cfg = copy.deepcopy(scheduler_cfg)
        schedulers.append(
            PARAM_SCHEDULERS.build(
                scheduler_cfg,
                default_args=dict(optimizer=optim_wrapper, epoch_length=epoch_length),
            ))
    return schedulers


def manual_train(cfg: Config, args: argparse.Namespace) -> None:
    ensure_import_path(args.config)
    from mmdet.utils import register_all_modules
    from mmdet.registry import MODELS

    register_all_modules(init_default_scope=True)

    train_loader = Runner.build_dataloader(cfg.train_dataloader, seed=args.seed)
    val_loader = Runner.build_dataloader(cfg.val_dataloader, seed=args.seed) if hasattr(cfg, "val_dataloader") else None

    model = MODELS.build(cfg.model)
    if hasattr(model, "init_weights"):
        model.init_weights()
    model = model.cuda()
    optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
    param_schedulers = build_param_schedulers(cfg.get("param_scheduler"), optim_wrapper, len(train_loader))
    message_hub = MessageHub.get_current_instance()
    model.train()
    total_epochs = int(cfg.train_cfg.get("max_epochs", 0))
    val_interval = int(cfg.train_cfg.get("val_interval", 1))
    val_begin = int(cfg.train_cfg.get("val_begin", 1))
    log_interval = int(cfg.default_hooks.get("logger", {}).get("interval", 50))
    checkpoint_cfg = cfg.default_hooks.get("checkpoint", {})
    checkpoint_interval = int(checkpoint_cfg.get("interval", 1))
    max_keep_ckpts = int(checkpoint_cfg.get("max_keep_ckpts", 5))
    save_best_key = str(checkpoint_cfg.get("save_best", "coco/bbox_mAP"))
    save_best_filename = f"best_{save_best_key.replace('/', '_')}.pth"

    early_stop_cfg = None
    for hook_cfg in cfg.get("custom_hooks", []):
        if hook_cfg.get("type") == "EarlyStoppingHook":
            early_stop_cfg = hook_cfg
            break
    patience = int(early_stop_cfg.get("patience", 0)) if early_stop_cfg else 0
    min_delta = float(early_stop_cfg.get("min_delta", 0.0)) if early_stop_cfg else 0.0
    rule = str(early_stop_cfg.get("rule", "greater")) if early_stop_cfg else "greater"

    work_dir = Path(cfg.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    resume_from = args.resume_from.resolve() if args.resume_from else None
    if resume_from is None and cfg.get("resume", False):
        candidate = work_dir / "last.pth"
        if candidate.exists():
            resume_from = candidate

    train_log_path = work_dir / "train.log"
    if resume_from is None:
        train_log_path.write_text("", encoding="utf-8")
    else:
        train_log_path.touch()
    MMLogger.get_instance("manual-train", log_file=str(train_log_path), log_level="INFO")

    manifest = make_manifest(args, cfg)
    if resume_from is not None:
        manifest["resume_from"] = str(resume_from)
    (work_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_iters = total_epochs * len(train_loader)
    global_iter = 0
    start_epoch = 1
    iter_times: list[float] = []
    best_value: float | None = None
    best_epoch: int | None = None
    best_metrics: dict | None = None
    latest_metrics: dict | None = None
    stale_epochs = 0

    eval_metrics_path = work_dir / "eval_metrics.json"
    if eval_metrics_path.exists():
        try:
            previous_eval = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
            best_value = previous_eval.get("best_value")
            best_epoch = previous_eval.get("best_epoch")
            best_metrics = previous_eval.get("best")
            latest_metrics = previous_eval.get("latest")
        except Exception:
            pass

    if resume_from is not None and resume_from.exists():
        checkpoint = torch.load(str(resume_from), map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        if "optimizer" in checkpoint:
            optim_wrapper.load_state_dict(checkpoint["optimizer"])
        ckpt_schedulers = checkpoint.get("param_schedulers", [])
        for scheduler, scheduler_state in zip(flatten_schedulers(param_schedulers), ckpt_schedulers):
            scheduler.load_state_dict(scheduler_state)
        if "message_hub" in checkpoint:
            try:
                message_hub.load_state_dict(checkpoint["message_hub"])
            except Exception:
                pass
        ckpt_meta = checkpoint.get("meta", {})
        global_iter = int(ckpt_meta.get("iter", 0))
        start_epoch = int(ckpt_meta.get("epoch", 0)) + 1
        if best_epoch is not None and start_epoch > 1:
            stale_epochs = max(0, start_epoch - 1 - best_epoch)
        emit_log(train_log_path, f"Resumed from checkpoint: {resume_from}")

    optim_wrapper.initialize_count_status(model, global_iter, total_iters)

    emit_log(train_log_path, "Manual training loop active (Runner.train disabled)")
    emit_log(train_log_path, f"Total epochs: {total_epochs}  iters/epoch: {len(train_loader)}")
    if args.adaptive_guard:
        emit_log(
            train_log_path,
            "Adaptive guard enabled: "
            f"gpu_util<={args.guard_gpu_util_pct:.1f}% "
            f"cpu_util<={args.guard_cpu_util_pct:.1f}% "
            f"resume<={args.guard_resume_util_pct:.1f}% "
            f"temp<={args.guard_temp_c:.1f}C "
            f"gpu_mem<={args.guard_memory_pct:.1f}% "
            f"cooldown={args.guard_cooldown_sec:.1f}s")

    if start_epoch > total_epochs:
        emit_log(train_log_path, f"Nothing to do: start_epoch={start_epoch} > max_epochs={total_epochs}")
        return

    guard_state: dict[str, float] = {}
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        epoch_iter = 0
        for data_batch in train_loader:
            epoch_iter += 1
            global_iter += 1
            adaptive_guard_pause(args, train_log_path, guard_state)

            data_start = time.perf_counter()
            # data_batch has already been fetched from dataloader; keep the
            # field for status compatibility.
            data_time = time.perf_counter() - data_start
            iter_start = time.perf_counter()

            with optim_wrapper.optim_context(model):
                data = model.data_preprocessor(data_batch, training=True)
                losses_dict = model(**data, mode="loss")
                parsed_loss, log_vars = model.parse_losses(losses_dict)

            optim_wrapper.update_params(parsed_loss)
            torch.cuda.synchronize()
            iter_time = time.perf_counter() - iter_start
            iter_times.append(iter_time)

            step_schedulers(param_schedulers, by_epoch=False)

            if epoch_iter % log_interval == 0 or epoch_iter == len(train_loader):
                remaining_iters = total_iters - global_iter
                eta = format_duration((sum(iter_times) / max(len(iter_times), 1)) * remaining_iters)
                grad_norm = get_scalar_float(message_hub, "train/grad_norm")
                memory_mb = int(torch.cuda.max_memory_allocated() / (1024 ** 2))
                loss_val = float(log_vars.get("loss", parsed_loss.detach().cpu()))
                loss_cls = float(log_vars.get("loss_cls", 0.0))
                loss_bbox = float(log_vars.get("loss_bbox", 0.0))
                emit_log(
                    train_log_path,
                    "Epoch(train)   "
                    f"[{epoch}][{epoch_iter:4d}/{len(train_loader)}]  "
                    f"lr: {first_optimizer_lr(optim_wrapper):.4e}  "
                    f"eta: {eta}  "
                    f"time: {iter_time:.4f}  "
                    f"data_time: {data_time:.4f}  "
                    f"memory: {memory_mb}  "
                    f"grad_norm: {0.0 if grad_norm is None else grad_norm:.4f}  "
                    f"loss: {loss_val:.4f}  "
                    f"loss_cls: {loss_cls:.4f}  "
                    f"loss_bbox: {loss_bbox:.4f}")

        step_schedulers(param_schedulers, by_epoch=True)

        if checkpoint_interval > 0 and (epoch % checkpoint_interval == 0 or epoch == total_epochs):
            save_checkpoint(work_dir, model, optim_wrapper, param_schedulers, message_hub, epoch, global_iter, latest_metrics, f"epoch_{epoch}.pth")
            copy_checkpoint_payload(work_dir / f"epoch_{epoch}.pth", work_dir / "last.pth")
            prune_old_epoch_checkpoints(work_dir, max_keep_ckpts)
            torch.cuda.empty_cache()

        if val_loader is not None and epoch >= val_begin and (epoch % val_interval == 0 or epoch == total_epochs):
            latest_metrics = run_validation(model, val_loader, cfg, epoch, train_log_path)
            current_value = float(latest_metrics.get(save_best_key, 0.0))
            if is_better(current_value, best_value, rule, min_delta):
                best_value = current_value
                best_epoch = epoch
                best_metrics = latest_metrics
                stale_epochs = 0
                if checkpoint_interval > 0:
                    source = work_dir / f"epoch_{epoch}.pth"
                    if source.exists():
                        copy_checkpoint_payload(source, work_dir / save_best_filename)
            else:
                stale_epochs += 1

            eval_payload = {
                "status": "ok",
                "epoch": epoch,
                "iter": global_iter,
                "latest": latest_metrics,
                "best_epoch": best_epoch,
                "best": best_metrics,
                "best_key": save_best_key,
                "best_value": best_value,
                "manual_loop": True,
            }
            (work_dir / "eval_metrics.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")

            if patience > 0 and stale_epochs >= patience:
                emit_log(train_log_path, f"Early stopping triggered at epoch {epoch}")
                break

    if not (work_dir / "eval_metrics.json").exists():
        eval_payload = {
            "status": "ok",
            "epoch": epoch,
            "iter": global_iter,
            "latest": latest_metrics,
            "best_epoch": best_epoch,
            "best": best_metrics,
            "best_key": save_best_key,
            "best_value": best_value,
            "manual_loop": True,
        }
        (work_dir / "eval_metrics.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    local_safe_mode = os.environ.get("MAMBA_LOCAL_SAFE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    runtime_changes: list[str] = []
    if local_safe_mode:
        runtime_changes = apply_local_safe_runtime_args(args)
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

    ensure_import_path(args.config)
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

    cfg = Config.fromfile(str(args.config))
    if args.work_dir is not None:
        cfg.work_dir = str(args.work_dir.resolve())
    cfg.randomness = dict(seed=args.seed)
    if hasattr(cfg, "env_cfg"):
        cfg.env_cfg.setdefault("mp_cfg", {})["mp_start_method"] = "fork"

    if local_safe_mode:
        changes = runtime_changes + apply_local_safe_mode(cfg, args)
        print("Local safe mode active:")
        for change in changes:
            print(f"- {change}")

    manual_train(cfg, args)


if __name__ == "__main__":
    main()
