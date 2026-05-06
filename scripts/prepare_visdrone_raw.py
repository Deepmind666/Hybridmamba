#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


SPLIT_MAP = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract VisDrone archives into the repository raw-data layout.")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=sorted(SPLIT_MAP))
    parser.add_argument("--output-root", default=Path("C:/mamba/data/visdrone"), type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_extract(zip_path: Path, temp_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(temp_dir)


def main() -> None:
    args = parse_args()
    split_dir = args.output_root / args.split
    images_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"

    if split_dir.exists() and args.overwrite:
        shutil.rmtree(split_dir)

    temp_dir = args.output_root / "_tmp_extract" / args.split
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    safe_extract(args.zip, temp_dir)

    expected_root = temp_dir / SPLIT_MAP[args.split]
    if not expected_root.exists():
        expected_root = temp_dir

    candidate_images = expected_root / "images"
    candidate_annotations = expected_root / "annotations"
    if not candidate_images.exists():
        raise FileNotFoundError(f"Images directory not found in extracted archive: {candidate_images}")
    if not candidate_annotations.exists():
        raise FileNotFoundError(f"Annotations directory not found in extracted archive: {candidate_annotations}")

    split_dir.mkdir(parents=True, exist_ok=True)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if ann_dir.exists():
        shutil.rmtree(ann_dir)
    shutil.move(str(candidate_images), str(images_dir))
    shutil.move(str(candidate_annotations), str(ann_dir))
    shutil.rmtree(temp_dir, ignore_errors=True)

    image_count = sum(1 for _ in images_dir.glob("*"))
    ann_count = sum(1 for _ in ann_dir.glob("*.txt"))
    print(
        json.dumps(
            {
                "split": args.split,
                "images_dir": str(images_dir),
                "annotations_dir": str(ann_dir),
                "image_count": image_count,
                "annotation_count": ann_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
