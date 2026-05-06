#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TRAIN_HEADER_RE = re.compile(
    r"Epoch\(train\)\s+\[(?P<epoch>\d+)\]\[\s*(?P<iter>\d+)/(?P<total>\d+)\]"
)

VAL_RE = re.compile(
    r"Epoch\(val\)\s+\[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\].*?"
    r"coco/bbox_mAP:\s+(?P<bbox_mAP>[0-9.]+)\s+"
    r"coco/bbox_mAP_50:\s+(?P<bbox_mAP_50>[0-9.]+)\s+"
    r"coco/bbox_mAP_75:\s+(?P<bbox_mAP_75>[0-9.]+)\s+"
    r"coco/bbox_mAP_s:\s+(?P<bbox_mAP_s>[0-9.]+)\s+"
    r"coco/bbox_mAP_m:\s+(?P<bbox_mAP_m>[0-9.]+)\s+"
    r"coco/bbox_mAP_l:\s+(?P<bbox_mAP_l>[0-9.]+)"
)

VAL_KEYS = ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show YOLO-style status for the latest or a specified run.")
    parser.add_argument("--run-dir", type=Path, help="Run directory under artifacts/runs. Defaults to latest modified.")
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    return parser.parse_args()


def latest_run_dir(runs_root: Path) -> Path:
    candidates = [path for path in runs_root.iterdir() if path.is_dir() and (path / "RUN_MANIFEST.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No run directories with RUN_MANIFEST.json found under {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_log(run_dir: Path) -> Path:
    logs = sorted(run_dir.rglob("*.log"))
    if not logs:
        raise FileNotFoundError(f"No log file found under {run_dir}")
    return max(logs, key=lambda path: path.stat().st_mtime)


def find_scalars(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("scalars.json"))
    return candidates[-1] if candidates else None


def parse_log(log_path: Path) -> tuple[dict | None, dict | None]:
    last_train = None
    last_val = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            train_match = TRAIN_HEADER_RE.search(line)
            if train_match:
                groups = train_match.groupdict()
                def extract_float(key: str) -> float | None:
                    match = re.search(rf"{re.escape(key)}:\s+([-+0-9.eE]+|inf|nan)", line)
                    return float(match.group(1)) if match else None

                eta_match = re.search(r"eta:\s+(.*?)(?=\s+time:)", line)
                last_train = {
                    "epoch": int(groups["epoch"]),
                    "iter": int(groups["iter"]),
                    "total_iter": int(groups["total"]),
                    "lr": extract_float("lr"),
                    "eta": eta_match.group(1).strip() if eta_match else "",
                    "time": extract_float("time"),
                    "data_time": extract_float("data_time"),
                    "memory": int(extract_float("memory")) if extract_float("memory") is not None else None,
                    "grad_norm": extract_float("grad_norm"),
                    "loss": extract_float("loss"),
                    "loss_cls": extract_float("loss_cls"),
                    "loss_bbox": extract_float("loss_bbox"),
                    "raw": line.strip(),
                }
            val_match = VAL_RE.search(line)
            if val_match:
                groups = val_match.groupdict()
                last_val = {
                    "epoch": int(groups["epoch"]),
                    "iter": int(groups["iter"]),
                    "total_iter": int(groups["total"]),
                    **{key: float(groups[key]) for key in VAL_KEYS},
                    "raw": line.strip(),
                }
    return last_train, last_val


def parse_best_val(scalars_path: Path | None) -> dict | None:
    if scalars_path is None:
        return None
    best = None
    with scalars_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if "coco/bbox_mAP" not in record:
                continue
            current = {
                "epoch": int(record.get("epoch", record.get("step", -1))),
                "step": int(record.get("step", -1)),
                "bbox_mAP": float(record.get("coco/bbox_mAP", 0.0)),
                "bbox_mAP_50": float(record.get("coco/bbox_mAP_50", 0.0)),
                "bbox_mAP_75": float(record.get("coco/bbox_mAP_75", 0.0)),
                "bbox_mAP_s": float(record.get("coco/bbox_mAP_s", 0.0)),
                "bbox_mAP_m": float(record.get("coco/bbox_mAP_m", 0.0)),
                "bbox_mAP_l": float(record.get("coco/bbox_mAP_l", 0.0)),
            }
            if best is None or current["bbox_mAP"] > best["bbox_mAP"]:
                best = current
    return best


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}"


def line(label: str, value: str) -> str:
    return f"{label:<16} {value}"


def status_from_log(log_path: Path) -> str:
    age_sec = max(0.0, __import__("time").time() - log_path.stat().st_mtime)
    if age_sec < 180:
        return "RUNNING"
    if age_sec < 1200:
        return "RECENT"
    return "STALE/FINISHED"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = (repo_root / args.runs_root).resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir(runs_root)

    manifest_path = run_dir / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    log_path = latest_log(run_dir)
    scalars_path = find_scalars(run_dir)
    train, latest_val = parse_log(log_path)
    best_val = parse_best_val(scalars_path)

    print("=" * 72)
    print("HybridMambaDet Training Status")
    print("=" * 72)
    print(line("Run", run_dir.name))
    print(line("Status", status_from_log(log_path)))
    if manifest:
        print(line("Model", manifest.get("model", "")))
        print(line("Dataset", manifest.get("dataset", "")))
        print(line("Detector", manifest.get("detector", "")))
        print(line("Created", manifest.get("created_at", "")))
    print(line("Log", str(log_path)))
    print()

    if train is not None:
        epoch_progress = 100.0 * train["iter"] / max(train["total_iter"], 1)
        print("[Train]")
        print(line("Epoch", f'{train["epoch"]}  iter {train["iter"]}/{train["total_iter"]} ({epoch_progress:.1f}%)'))
        print(line("ETA", train["eta"]))
        print(line("LR", f'{train["lr"]:.6g}'))
        print(line("Loss", f'{train["loss"]:.4f}  cls {train["loss_cls"]:.4f}  bbox {train["loss_bbox"]:.4f}'))
        print(line("GradNorm", f'{train["grad_norm"]:.4f}'))
        print(line("IterTime", f'{train["time"]:.4f}s  data {train["data_time"]:.4f}s'))
        print(line("Memory", f'{train["memory"]} MB'))
    else:
        print("[Train]")
        print("No train iteration parsed yet.")
    print()

    print("[Validation]")
    if latest_val is not None:
        print(
            line(
                "Latest",
                f'epoch {latest_val["epoch"]}  '
                f'AP {format_pct(latest_val["bbox_mAP"])}  '
                f'AP50 {format_pct(latest_val["bbox_mAP_50"])}  '
                f'APs {format_pct(latest_val["bbox_mAP_s"])}',
            )
        )
        print(
            line(
                "Latest+",
                f'AP75 {format_pct(latest_val["bbox_mAP_75"])}  '
                f'APm {format_pct(latest_val["bbox_mAP_m"])}  '
                f'APl {format_pct(latest_val["bbox_mAP_l"])}',
            )
        )
    else:
        print("No validation metrics yet.")

    if best_val is not None:
        print(
            line(
                "Best",
                f'epoch {best_val["epoch"]}  '
                f'AP {format_pct(best_val["bbox_mAP"])}  '
                f'AP50 {format_pct(best_val["bbox_mAP_50"])}  '
                f'APs {format_pct(best_val["bbox_mAP_s"])}',
            )
        )
        print(
            line(
                "Best+",
                f'AP75 {format_pct(best_val["bbox_mAP_75"])}  '
                f'APm {format_pct(best_val["bbox_mAP_m"])}  '
                f'APl {format_pct(best_val["bbox_mAP_l"])}',
            )
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
