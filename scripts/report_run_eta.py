#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TRAIN_RE = re.compile(
    r"Epoch\(train\)\s+\[(?P<epoch>\d+)\]\[\s*(?P<iter>\d+)/(?P<total>\d+)\].*?eta:\s+(?P<eta>.*?)(?:\s+time:)"
)
VAL_RE = re.compile(
    r"Epoch\(val\)\s+\[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\].*?(coco/bbox_mAP:\s+(?P<map>[0-9.]+))?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report latest ETA and status from an MMEngine run directory.")
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def latest_log(run_dir: Path) -> Path:
    logs = sorted(run_dir.rglob("*.log"))
    if not logs:
        raise FileNotFoundError(f"No log file found under {run_dir}")
    return max(logs, key=lambda path: path.stat().st_mtime)


def main() -> None:
    args = parse_args()
    log_path = latest_log(args.run_dir.resolve())
    last_train = None
    last_val = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = TRAIN_RE.search(line)
            if match:
                last_train = {
                    "epoch": int(match.group("epoch")),
                    "iter": int(match.group("iter")),
                    "total_iter": int(match.group("total")),
                    "eta": match.group("eta"),
                    "raw": line.strip(),
                }
            match = VAL_RE.search(line)
            if match:
                last_val = {
                    "epoch": int(match.group("epoch")),
                    "iter": int(match.group("iter")),
                    "total_iter": int(match.group("total")),
                    "bbox_mAP": match.group("map"),
                    "raw": line.strip(),
                }

    report = {
        "run_dir": str(args.run_dir.resolve()),
        "log_path": str(log_path),
        "latest_train": last_train,
        "latest_val": last_val,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
