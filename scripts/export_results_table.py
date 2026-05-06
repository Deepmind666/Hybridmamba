#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_FIELDS = [
    "bbox_mAP",
    "bbox_mAP_50",
    "bbox_mAP_75",
    "bbox_mAP_s",
    "bbox_mAP_m",
    "bbox_mAP_l",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export eval json files into a flat CSV / Markdown table.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> Dict[str, object]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def collect_rows(input_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metrics_path in sorted(input_root.rglob("eval_metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = load_manifest(metrics_path.parent / "RUN_MANIFEST.json")
        row = {
            "run_dir": str(metrics_path.parent),
            "run_id": manifest.get("run_id", metrics_path.parent.name),
            "dataset": manifest.get("dataset", ""),
            "model": manifest.get("model", ""),
            "stack": manifest.get("stack", ""),
            "config": manifest.get("config", ""),
            "checkpoint": manifest.get("checkpoint", ""),
            "step": metrics.get("step", ""),
        }
        for field in DEFAULT_FIELDS:
            row[field] = metrics.get(field, "")
        # Skip placeholder smoke outputs that do not carry any extracted metrics.
        if not row["step"] and not any(row[field] not in ("", None) for field in DEFAULT_FIELDS):
            continue
        rows.append(row)
    return rows


def write_csv(output_path: Path, rows: List[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(output_path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fieldnames) + " |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.input_root)
    write_csv(args.output_csv, rows)
    if args.output_md:
        write_markdown(args.output_md, rows)
    print(json.dumps({"rows": len(rows), "output_csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
