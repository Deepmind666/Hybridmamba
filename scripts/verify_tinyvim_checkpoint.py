#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and optionally promote a TinyViM checkpoint candidate.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--promote-to", type=Path)
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_with_torch(path: Path) -> dict:
    try:
        import torch
    except Exception as exc:
        return {"torch_available": False, "error": f"{type(exc).__name__}: {exc}"}

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    top_keys = sorted(payload.keys()) if isinstance(payload, dict) else []
    model_keys = sorted(payload["model"].keys())[:20] if isinstance(payload, dict) and "model" in payload else []
    return {
        "torch_available": True,
        "payload_type": type(payload).__name__,
        "top_level_keys": top_keys,
        "has_model": isinstance(payload, dict) and "model" in payload,
        "model_key_preview": model_keys,
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    report = {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": sha256sum(checkpoint),
        "is_zipfile": zipfile.is_zipfile(checkpoint),
    }
    if report["is_zipfile"]:
        with zipfile.ZipFile(checkpoint) as archive:
            names = archive.namelist()
            report["zip_entries"] = len(names)
            report["zip_preview"] = names[:10]

    report["torch_check"] = inspect_with_torch(checkpoint)

    promotable = report["is_zipfile"] and report["torch_check"].get("has_model", False)
    report["promotable"] = promotable

    if promotable and args.promote_to:
        target = args.promote_to.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target)
        report["promoted_to"] = str(target)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
