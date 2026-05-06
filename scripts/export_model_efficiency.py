#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "analysis" / "model_efficiency"
DEFAULT_CONFIGS = [
    (
        "TinyViM-B",
        ROOT / "code" / "tinyvim" / "detection" / "configs_v3" / "retinanet_tinyvim_b_fpn_1x_visdrone_es_stable.py",
    ),
    (
        "HybridMamba-Base",
        ROOT
        / "code"
        / "tinyvim"
        / "detection"
        / "configs_v3"
        / "retinanet_hybridmamba_base_b_fpn_1x_visdrone_es_stable.py",
    ),
    (
        "HybridMambaDet",
        ROOT
        / "code"
        / "tinyvim"
        / "detection"
        / "configs_v3"
        / "retinanet_hybridmambadet_b_fpn_1x_visdrone_es_stable.py",
    ),
    (
        "Fusion alpha=0.5",
        ROOT
        / "code"
        / "tinyvim"
        / "detection"
        / "configs_v3"
        / "retinanet_hybridmambadet_b_fpn_1x_visdrone_es_fusion05_stable.py",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export parameter count and extract_feat throughput for MMDet3 configs."
    )
    parser.add_argument(
        "--config",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Model label and config path. May be repeated. Defaults to the four paper models.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=int(os.environ.get("TORCH_NUM_THREADS", "1")))
    return parser.parse_args()


def detection_root(config_path: Path) -> Path:
    return config_path.resolve().parents[1]


def ensure_import_path(config_path: Path) -> None:
    root = detection_root(config_path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    import model  # noqa: F401


def count_params(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters())


def build_detector(config_path: Path) -> tuple[torch.nn.Module, Config]:
    ensure_import_path(config_path)
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    model = MODELS.build(cfg.model)
    model.eval()
    return model, cfg


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_extract_feat(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    height: int,
    width: int,
    warmup_iters: int,
    benchmark_iters: int,
) -> dict[str, float]:
    dummy = torch.randn(batch_size, 3, height, width, device=device)
    model = model.to(device)

    with torch.inference_mode():
        for _ in range(max(0, warmup_iters)):
            _ = model.extract_feat(dummy)
        synchronize(device)

        timings: list[float] = []
        for _ in range(max(1, benchmark_iters)):
            start = time.perf_counter()
            _ = model.extract_feat(dummy)
            synchronize(device)
            timings.append(time.perf_counter() - start)

    per_image = [elapsed / batch_size for elapsed in timings]
    return {
        "latency_ms_mean": mean(per_image) * 1000.0,
        "latency_ms_median": median(per_image) * 1000.0,
        "throughput_img_s": batch_size / mean(timings),
    }


def profile_flops(model: torch.nn.Module, device: torch.device, batch_size: int, height: int, width: int) -> int | None:
    if device.type != "cuda":
        return None
    dummy = torch.randn(batch_size, 3, height, width, device=device)
    model = model.to(device)
    with torch.inference_mode():
        try:
            with torch.profiler.profile(with_flops=True, activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                _ = model.extract_feat(dummy)
                synchronize(device)
            return int(sum(event.flops for event in prof.key_averages() if event.flops is not None))
        except Exception:
            return None


def config_pairs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.config:
        return [(label, Path(path).resolve()) for label, path in args.config]
    return [(label, path.resolve()) for label, path in DEFAULT_CONFIGS]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0].keys())
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_num_threads))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    rows: list[dict[str, Any]] = []
    for label, config_path in config_pairs(args):
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        model, cfg = build_detector(config_path)
        model_type = cfg.model.get("backbone", {}).get("type", "")
        row: dict[str, Any] = {
            "label": label,
            "config": str(config_path),
            "backbone": model_type,
            "input_hw": f"{args.height}x{args.width}",
            "batch_size": args.batch_size,
            "full_params_m": round(count_params(model) / 1_000_000.0, 4),
            "backbone_params_m": round(count_params(getattr(model, "backbone", None)) / 1_000_000.0, 4),
            "neck_params_m": round(count_params(getattr(model, "neck", None)) / 1_000_000.0, 4),
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
            "status": "ok",
            "flops_profiled_g": "",
            "flops_note": "torch profiler FLOPs are partial; custom selective-scan ops may be uncounted",
        }
        try:
            row.update(
                {
                    key: round(value, 4)
                    for key, value in benchmark_extract_feat(
                        model,
                        device,
                        args.batch_size,
                        args.height,
                        args.width,
                        args.warmup_iters,
                        args.benchmark_iters,
                    ).items()
                }
            )
            if args.profile_flops:
                flops = profile_flops(model, device, args.batch_size, args.height, args.width)
                row["flops_profiled_g"] = "" if flops is None else round(flops / 1_000_000_000.0, 4)
        except Exception as exc:
            row["status"] = f"failed: {type(exc).__name__}: {exc}"
            row["latency_ms_mean"] = ""
            row["latency_ms_median"] = ""
            row["throughput_img_s"] = ""
        finally:
            rows.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_efficiency.csv"
    md_path = output_dir / "model_efficiency.md"
    json_path = output_dir / "model_efficiency.json"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
