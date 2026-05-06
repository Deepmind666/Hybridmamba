#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize AI-TOD / AI-TOD-v2 annotations into contiguous COCO ids.")
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-area", type=float, default=1.0)
    return parser.parse_args()


def remap_categories(coco: Dict[str, object], min_area: float) -> Dict[str, object]:
    categories = coco.get("categories", [])
    if not categories:
        raise ValueError("Input annotation file does not contain categories.")

    category_map = {
        int(category["id"]): new_id
        for new_id, category in enumerate(sorted(categories, key=lambda item: int(item["id"])), start=1)
    }
    remapped_categories: List[Dict[str, object]] = []
    for category in sorted(categories, key=lambda item: int(item["id"])):
        category = dict(category)
        category["id"] = category_map[int(category["id"])]
        remapped_categories.append(category)

    remapped_annotations: List[Dict[str, object]] = []
    next_annotation_id = 1
    for annotation in coco.get("annotations", []):
        bbox = annotation.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        area = float(annotation.get("area", bbox[2] * bbox[3]))
        if bbox[2] <= 0 or bbox[3] <= 0 or area < min_area:
            continue
        annotation = dict(annotation)
        annotation["id"] = next_annotation_id
        annotation["category_id"] = category_map[int(annotation["category_id"])]
        annotation["area"] = area
        annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
        remapped_annotations.append(annotation)
        next_annotation_id += 1

    return {
        "info": coco.get("info", {"description": "AI-TOD normalized to COCO"}),
        "licenses": coco.get("licenses", []),
        "images": coco.get("images", []),
        "annotations": remapped_annotations,
        "categories": remapped_categories,
    }


def main() -> None:
    args = parse_args()
    coco = json.loads(args.input_json.read_text(encoding="utf-8"))
    normalized = remap_categories(coco, min_area=args.min_area)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "images": len(normalized["images"]),
                "annotations": len(normalized["annotations"]),
                "categories": len(normalized["categories"]),
                "output_json": str(args.output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

