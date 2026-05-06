#!/usr/bin/env python
"""Materialize ImageNet-1K ImageFolder layout from official ILSVRC2012 archives."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path


def _ensure_devkit_extracted(root: Path) -> Path:
    devkit_root = root / "ILSVRC2012_devkit_t12"
    meta_path = devkit_root / "data" / "meta.mat"
    if meta_path.exists():
        return meta_path
    devkit_tar = root / "ILSVRC2012_devkit_t12.tar.gz"
    if not devkit_tar.exists():
        raise FileNotFoundError(devkit_tar)
    with tarfile.open(devkit_tar) as tar:
        for member in tar.getmembers():
            _safe_extract_member(tar, member, root)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return meta_path


def _load_val_wnids_from_devkit(root: Path) -> list[str]:
    import numpy as np
    from scipy.io import loadmat

    meta_path = _ensure_devkit_extracted(root)
    gt_path = meta_path.parent / "ILSVRC2012_validation_ground_truth.txt"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    mat = loadmat(meta_path, squeeze_me=True, struct_as_record=False)
    id_to_wnid: dict[int, str] = {}
    for synset in np.ravel(mat["synsets"]):
        num_children = int(getattr(synset, "num_children"))
        if num_children != 0:
            continue
        class_id = int(getattr(synset, "ILSVRC2012_ID"))
        wnid = str(getattr(synset, "WNID"))
        id_to_wnid[class_id] = wnid

    labels = [int(line.strip()) for line in gt_path.read_text(encoding="ascii").splitlines() if line.strip()]
    return [id_to_wnid[label] for label in labels]


def _safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    target = (dest / member.name).resolve()
    dest_resolved = dest.resolve()
    if dest_resolved != target and dest_resolved not in target.parents:
        raise RuntimeError(f"Refusing unsafe tar member path: {member.name}")
    tar.extract(member, dest)


def extract_train(root: Path, force: bool = False) -> dict[str, int]:
    train_root = root / "train"
    train_tar_path = root / "train.tar"
    if not train_tar_path.exists():
        raise FileNotFoundError(train_tar_path)
    train_root.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}
    with tarfile.open(train_tar_path) as train_tar:
        members = [m for m in train_tar.getmembers() if m.isfile() and m.name.endswith(".tar")]
        for member in members:
            wnid = Path(member.name).stem
            class_dir = train_root / wnid
            done_marker = class_dir / ".complete"
            if done_marker.exists() and not force:
                stats[wnid] = len(list(class_dir.glob("*.JPEG")))
                continue

            class_dir.mkdir(parents=True, exist_ok=True)
            nested_tar_path = train_tar.extractfile(member)
            if nested_tar_path is None:
                raise RuntimeError(f"Could not read nested train archive: {member.name}")
            with tarfile.open(fileobj=nested_tar_path) as class_tar:
                for class_member in class_tar.getmembers():
                    if class_member.isfile():
                        _safe_extract_member(class_tar, class_member, class_dir)
            done_marker.write_text("ok\n", encoding="ascii")
            stats[wnid] = len(list(class_dir.glob("*.JPEG")))
    return stats


def extract_val(root: Path, force: bool = False) -> dict[str, int]:
    val_root = root / "val"
    val_tar_path = root / "val.tar"
    if not val_tar_path.exists():
        raise FileNotFoundError(val_tar_path)
    val_wnids = _load_val_wnids_from_devkit(root)
    if len(val_wnids) != 50000:
        raise RuntimeError(f"Expected 50000 validation labels, got {len(val_wnids)}")

    val_done = val_root / ".complete"
    if val_done.exists() and not force:
        return {p.name: len(list(p.glob("*.JPEG"))) for p in val_root.iterdir() if p.is_dir()}

    staging = root / "_val_flat_tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(val_tar_path) as val_tar:
        val_tar.extractall(staging)

    files = sorted(staging.glob("ILSVRC2012_val_*.JPEG"))
    if len(files) != 50000:
        raise RuntimeError(f"Expected 50000 validation images, got {len(files)}")

    stats: dict[str, int] = {}
    for image_path, wnid in zip(files, val_wnids):
        class_dir = val_root / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / image_path.name
        if dest.exists() and force:
            dest.unlink()
        if not dest.exists():
            shutil.move(str(image_path), str(dest))
        stats[wnid] = stats.get(wnid, 0) + 1

    shutil.rmtree(staging)
    val_done.write_text("ok\n", encoding="ascii")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"C:\mamba\data\imagenet")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    summary: dict[str, object] = {"root": str(root)}
    if args.split in {"train", "both"}:
        train_stats = extract_train(root, force=args.force)
        summary["train_classes"] = len(train_stats)
        summary["train_images"] = sum(train_stats.values())
    if args.split in {"val", "both"}:
        val_stats = extract_val(root, force=args.force)
        summary["val_classes"] = len(val_stats)
        summary["val_images"] = sum(val_stats.values())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
