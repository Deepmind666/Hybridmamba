#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a file from Google Drive with confirm-token handling.")
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={"id": args.file_id}, stream=True, timeout=60)
    response.raise_for_status()

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(url, params={"id": args.file_id, "confirm": token}, stream=True, timeout=60)
        response.raise_for_status()
    else:
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            html = response.text
            if "download-form" in html:
                action_match = re.search(r'<form id="download-form" action="([^"]+)"', html)
                hidden_inputs = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', html))
                if action_match and hidden_inputs:
                    action = action_match.group(1)
                    response = session.get(action, params=hidden_inputs, stream=True, timeout=60)
                    response.raise_for_status()

    total_bytes = 0
    with args.output.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=args.chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
            total_bytes += len(chunk)

    print(json.dumps({"output": str(args.output), "bytes": total_bytes}, indent=2))


if __name__ == "__main__":
    main()
