#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch an installed MMDetection 3.x package to accept MMCV 2.2.0.")
    parser.add_argument("--site-packages", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_path = args.site_packages / "mmdet" / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(init_path)

    text = init_path.read_text(encoding="utf-8")
    original = text
    text = text.replace("mmcv_maximum_version = '2.2.0'", "mmcv_maximum_version = '2.3.0'")
    changed = text != original
    if changed:
        init_path.write_text(text, encoding="utf-8")

    print(
        json.dumps(
            {
                "init_path": str(init_path),
                "patched": changed,
                "target_line": "mmcv_maximum_version = '2.3.0'",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
