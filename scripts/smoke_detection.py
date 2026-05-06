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
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import get_root_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a short single-GPU detector smoke pass.')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--max-iters', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=int, default=0)
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
    run_id = f"smoke_{args.config.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    backbone_cfg = cfg.model.get('backbone', {})
    init_cfg = backbone_cfg.get('init_cfg', {})
    return {
        'run_id': run_id,
        'config': str(args.config.resolve()),
        'work_dir': str(args.work_dir.resolve()),
        'dataset_config': args.config.stem,
        'backbone_type': backbone_cfg.get('type', ''),
        'backbone_init_checkpoint': init_cfg.get('checkpoint', ''),
        'max_iters': args.max_iters,
        'seed': args.seed,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'device_capability': torch.cuda.get_device_capability(args.gpu_id) if torch.cuda.is_available() else None,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the TinyViM detector smoke test.')
    try:
        _ = torch.randn(1, device=f'cuda:{args.gpu_id}')
    except Exception as exc:
        raise RuntimeError(
            'CUDA is visible but unusable for the current PyTorch build. '
            'See docs/runtime-blockers.md before attempting detector smoke.'
        ) from exc

    cfg = Config.fromfile(str(args.config))
    ensure_import_path(args.config)

    cfg.work_dir = str(args.work_dir.resolve())
    cfg.gpu_ids = [args.gpu_id]
    cfg.seed = args.seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(args, cfg)
    (args.work_dir / 'RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    log_file = args.work_dir / 'train.log'
    logger = get_root_logger(log_file=str(log_file), log_level='INFO')
    logger.info('Starting smoke run for %s', args.config)

    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=args.seed,
    )

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'),
    )
    model.init_weights()
    if hasattr(dataset, 'CLASSES'):
        model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=[args.gpu_id])
    optimizer = build_optimizer(model, cfg.optimizer)

    model.train()
    timings = []
    losses = []
    data_iter = iter(data_loader)
    for step in range(1, args.max_iters + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(return_loss=True, **batch)
        loss, _ = model.module._parse_losses(loss_dict)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(args.gpu_id)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        losses.append(float(loss.detach().cpu()))
        logger.info('iter=%d loss=%.6f time=%.4fs', step, losses[-1], elapsed)

    if not losses:
        raise RuntimeError('Smoke run did not consume any batch.')

    metrics = {
        'status': 'ok',
        'iters': len(losses),
        'mean_loss': mean(losses),
        'last_loss': losses[-1],
        'mean_iter_time_sec': mean(timings),
        'cuda_device': torch.cuda.get_device_name(args.gpu_id),
    }
    (args.work_dir / 'eval_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
