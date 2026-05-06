#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a COCO annotation file.")
    parser.add_argument("--input-json", required=True, type=Path)
    return parser.parse_args()


def bucket_name(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def main() -> None:
    args = parse_args()
    coco = json.loads(args.input_json.read_text(encoding="utf-8"))
    category_lookup = {int(category["id"]): category["name"] for category in coco.get("categories", [])}

    size_counter = Counter()
    category_counter = Counter()
    for annotation in coco.get("annotations", []):
        area = float(annotation.get("area", 0.0))
        size_counter[bucket_name(area)] += 1
        category_counter[category_lookup.get(int(annotation["category_id"]), str(annotation["category_id"]))] += 1

    summary = {
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "categories": len(coco.get("categories", [])),
        "size_breakdown": dict(size_counter),
        "category_breakdown": dict(sorted(category_counter.items())),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

