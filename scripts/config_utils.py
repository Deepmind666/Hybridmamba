#!/usr/bin/env python3
from __future__ import annotations

import copy
import runpy
from pathlib import Path
from typing import Any, Dict


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            if key == "_delete_" and value:
                return {k: copy.deepcopy(v) for k, v in override.items() if k != "_delete_"}
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def load_python_config(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.resolve()
    namespace = runpy.run_path(str(config_path))

    merged: Dict[str, Any] = {}
    for base_entry in namespace.get("_base_", []):
        base_path = (config_path.parent / base_entry).resolve()
        merged = deep_merge(merged, load_python_config(base_path))

    local_entries = {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__") and key != "_base_"
    }
    merged = deep_merge(merged, local_entries)
    merged["_config_path"] = str(config_path)
    return merged


def detection_root_from_config(config_path: Path) -> Path:
    return config_path.resolve().parents[1]


def resolve_runtime_path(config_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (detection_root_from_config(config_path) / candidate).resolve()

