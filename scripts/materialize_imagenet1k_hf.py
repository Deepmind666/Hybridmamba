#!/usr/bin/env python
"""Materialize Hugging Face ImageNet-1K parquet shards as ImageFolder.

This creates the directory layout expected by the TinyViM and MobileMamba
launchers:

    output_root/
      train/<label_id>/*.JPEG
      val/<label_id>/*.JPEG

The Hugging Face dataset is gated. A valid token with accepted access is
required through --token or the HF_TOKEN/HUGGING_FACE_HUB_TOKEN environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def split_to_output_name(split: str) -> str:
    return "val" if split == "validation" else split


def normalize_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def iter_splits(raw_splits: str) -> Iterable[str]:
    for item in raw_splits.replace(",", " ").split():
        item = item.strip()
        if item:
            yield item


def materialize_split(args: argparse.Namespace, split: str) -> None:
    out_name = split_to_output_name(split)
    out_root = Path(args.output_root) / out_name
    out_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.repo_id,
        split=split,
        cache_dir=args.cache_dir,
        token=args.token or None,
        trust_remote_code=False,
    )

    features = dataset.features
    label_feature = features.get("label")
    label_count = len(getattr(label_feature, "names", []) or []) or args.num_classes
    for label_id in range(label_count):
        (out_root / f"{label_id:04d}").mkdir(parents=True, exist_ok=True)

    progress = tqdm(dataset, desc=f"materialize {split}", unit="img")
    written = 0
    skipped = 0
    for index, sample in enumerate(progress):
        label = int(sample["label"])
        class_dir = out_root / f"{label:04d}"
        suffix = args.image_suffix
        dst = class_dir / f"{out_name}_{index:08d}{suffix}"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        image = normalize_image(sample["image"])
        image.save(dst, quality=args.jpeg_quality)
        written += 1
        if args.max_samples and written >= args.max_samples:
            break

    marker = out_root / "_materialize_complete.txt"
    marker.write_text(
        f"repo_id={args.repo_id}\nsplit={split}\nwritten={written}\nskipped={skipped}\n",
        encoding="ascii",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--output-root", default=r"C:\mamba\data\imagenet")
    parser.add_argument("--cache-dir", default=r"C:\mamba\data\downloads\hf_cache")
    parser.add_argument("--splits", default="train validation")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--image-suffix", default=".JPEG")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit(
            "Missing Hugging Face token. Run `hf auth login` or pass --token after accepting access to ILSVRC/imagenet-1k."
        )
    for split in iter_splits(args.splits):
        materialize_split(args, split)


if __name__ == "__main__":
    main()
