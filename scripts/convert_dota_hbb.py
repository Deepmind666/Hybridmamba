#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


DOTA_CATEGORIES = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
]
CATEGORY_TO_ID = {name: index for index, name in enumerate(DOTA_CATEGORIES, start=1)}


@dataclass
class DotaObject:
    category: str
    difficulty: int
    bbox: Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DOTA OBB labels into patch-level HBB COCO annotations.")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=824)
    parser.add_argument("--min-cover", type=float, default=0.5)
    parser.add_argument("--difficulty-threshold", type=int, default=1)
    parser.add_argument("--keep-empty", action="store_true")
    return parser.parse_args()


def iter_images(images_dir: Path) -> Iterable[Path]:
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp"):
        for image_path in sorted(images_dir.glob(suffix)):
            yield image_path


def parse_label_file(label_path: Path, difficulty_threshold: int) -> List[DotaObject]:
    objects: List[DotaObject] = []
    if not label_path.exists():
        return objects

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line or raw_line.startswith("imagesource") or raw_line.startswith("gsd"):
            continue
        parts = raw_line.split()
        if len(parts) < 9:
            continue
        coords = [float(value) for value in parts[:8]]
        category = parts[8]
        difficulty = int(parts[9]) if len(parts) > 9 else 0
        if difficulty > difficulty_threshold or category not in CATEGORY_TO_ID:
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        objects.append(
            DotaObject(
                category=category,
                difficulty=difficulty,
                bbox=(min_x, min_y, max_x - min_x, max_y - min_y),
            )
        )
    return objects


def generate_starts(length: int, crop_size: int, stride: int) -> List[int]:
    if length <= crop_size:
        return [0]
    starts = list(range(0, length - crop_size + 1, stride))
    if starts[-1] != length - crop_size:
        starts.append(length - crop_size)
    return starts


def intersect_bbox(bbox: Tuple[float, float, float, float], patch: Tuple[int, int, int, int]) -> Tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    px0, py0, px1, py1 = patch
    bx0, by0, bx1, by1 = x, y, x + w, y + h
    ix0, iy0 = max(bx0, px0), max(by0, py0)
    ix1, iy1 = min(bx1, px1), min(by1, py1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, iy0, ix1 - ix0, iy1 - iy0


def build_coco(args: argparse.Namespace) -> Dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    image_id = 1
    annotation_id = 1

    for image_path in iter_images(args.images_dir):
        label_path = args.label_dir / f"{image_path.stem}.txt"
        objects = parse_label_file(label_path, args.difficulty_threshold)
        with Image.open(image_path) as image:
            width, height = image.size
            x_starts = generate_starts(width, args.crop_size, args.stride)
            y_starts = generate_starts(height, args.crop_size, args.stride)

            for y0 in y_starts:
                for x0 in x_starts:
                    x1 = min(width, x0 + args.crop_size)
                    y1 = min(height, y0 + args.crop_size)
                    patch = (x0, y0, x1, y1)
                    patch_annotations: List[Dict[str, object]] = []

                    for obj in objects:
                        clipped = intersect_bbox(obj.bbox, patch)
                        if clipped is None:
                            continue
                        original_area = obj.bbox[2] * obj.bbox[3]
                        clipped_area = clipped[2] * clipped[3]
                        if original_area <= 0 or clipped_area / original_area < args.min_cover:
                            continue
                        patch_annotations.append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": CATEGORY_TO_ID[obj.category],
                                "bbox": [clipped[0] - x0, clipped[1] - y0, clipped[2], clipped[3]],
                                "area": clipped_area,
                                "iscrowd": 0,
                                "difficulty": obj.difficulty,
                            }
                        )
                        annotation_id += 1

                    if not patch_annotations and not args.keep_empty:
                        continue

                    patch_name = f"{image_path.stem}_{x0}_{y0}_{x1}_{y1}{image_path.suffix.lower()}"
                    patch_image_path = args.output_dir / patch_name
                    if not patch_image_path.exists():
                        cropped = image.crop((x0, y0, x1, y1))
                        cropped.save(patch_image_path)

                    images.append(
                        {
                            "id": image_id,
                            "file_name": patch_name,
                            "width": x1 - x0,
                            "height": y1 - y0,
                            "source_image": image_path.name,
                        }
                    )
                    annotations.extend(patch_annotations)
                    image_id += 1

    return {
        "info": {"description": "DOTA v1.0 converted to HBB COCO patches"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_id, "name": name} for name, category_id in CATEGORY_TO_ID.items()],
    }


def main() -> None:
    args = parse_args()
    coco = build_coco(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "images": len(coco["images"]),
                "annotations": len(coco["annotations"]),
                "categories": len(coco["categories"]),
                "patch_dir": str(args.output_dir),
                "output_json": str(args.output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

