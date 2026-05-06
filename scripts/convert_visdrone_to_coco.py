#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image


VISDRONE_CATEGORIES = [
    (1, "pedestrian"),
    (2, "people"),
    (3, "bicycle"),
    (4, "car"),
    (5, "van"),
    (6, "truck"),
    (7, "tricycle"),
    (8, "awning-tricycle"),
    (9, "bus"),
    (10, "motor"),
]
VALID_CATEGORY_IDS = {category_id for category_id, _ in VISDRONE_CATEGORIES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET annotations to COCO.")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--annotations-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def iter_images(images_dir: Path) -> Iterable[Path]:
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for path in sorted(images_dir.glob(suffix)):
            yield path


def parse_annotation_line(line: str) -> List[float]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 8:
        raise ValueError(f"Unexpected annotation format: {line}")
    return [float(part) for part in parts[:8]]


def build_coco(images_dir: Path, annotations_dir: Path, min_area: float, allow_empty: bool) -> Dict[str, object]:
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    for image_path in iter_images(images_dir):
        with Image.open(image_path) as image:
            width, height = image.size

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        valid_for_image = 0
        if annotation_path.exists():
            for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                x, y, w, h, score, category_id, truncation, occlusion = parse_annotation_line(raw_line)
                category_id = int(category_id)
                if category_id not in VALID_CATEGORY_IDS:
                    continue
                if w <= 0 or h <= 0 or (w * h) < min_area:
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "truncation": truncation,
                        "occlusion": occlusion,
                        "score": score,
                    }
                )
                annotation_id += 1
                valid_for_image += 1

        if valid_for_image == 0 and not allow_empty:
            images.pop()
            image_id += 1
            continue

        image_id += 1

    return {
        "info": {
            "description": "VisDrone DET converted to COCO",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_id, "name": name} for category_id, name in VISDRONE_CATEGORIES],
    }


def main() -> None:
    args = parse_args()
    coco = build_coco(args.images_dir, args.annotations_dir, args.min_area, args.allow_empty)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "images": len(coco["images"]),
                "annotations": len(coco["annotations"]),
                "categories": len(coco["categories"]),
                "output_json": str(args.output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

