#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh eval_metrics.json for all MMDet3-style runs.")
    parser.add_argument("--runs-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).resolve().parent / "extract_mmdet3_metrics.py"
    updated = []
    for run_dir in sorted(args.runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest = run_dir / "RUN_MANIFEST.json"
        if not manifest.exists():
            continue
        try:
            subprocess.run(
                [sys.executable, str(script), "--run-dir", str(run_dir), "--output-json", str(run_dir / "eval_metrics.json")],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            updated.append(str(run_dir))
        except Exception:
            continue
    print(json.dumps({"updated_runs": updated, "count": len(updated)}, indent=2))


if __name__ == "__main__":
    main()
