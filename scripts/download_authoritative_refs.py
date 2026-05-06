#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download authoritative paper/reference assets from the local manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("C:/mamba/references/reference_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("C:/mamba/references/papers"))
    return parser.parse_args()


def download_binary(url: str, output_path: Path) -> int:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = 0
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                total += len(chunk)
    return total


def download_text(url: str, output_path: Path) -> int:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")
    return len(response.content)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for item in manifest:
        target = args.output_dir / item["filename"]
        url = item["pdf_url"] or item["source_url"]
        if not url:
            continue
        try:
            if target.suffix.lower() in {".html", ".txt"}:
                num_bytes = download_text(url, target)
            else:
                num_bytes = download_binary(url, target)
            downloaded.append({"key": item["key"], "output": str(target), "bytes": num_bytes})
        except Exception as exc:
            downloaded.append({"key": item["key"], "output": str(target), "error": f"{type(exc).__name__}: {exc}"})

    print(json.dumps({"downloaded": downloaded, "count": len(downloaded)}, indent=2))


if __name__ == "__main__":
    main()

