#!/usr/bin/env python3
from __future__ import annotations

import json


def main() -> None:
    report = {}
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
            report["cuda_capability"] = torch.cuda.get_device_capability(0)
            try:
                x = torch.randn(1, device="cuda")
                report["cuda_tensor_ok"] = float(x.item())
            except Exception as exc:
                report["cuda_tensor_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"

    for module_name in ("mmcv", "mmdet", "timm", "einops", "cv2", "numpy", "setuptools"):
        try:
            module = __import__(module_name)
            report[module_name] = getattr(module, "__version__", "n/a")
        except Exception as exc:
            report[f"{module_name}_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

