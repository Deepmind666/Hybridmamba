#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RunRecord:
    dataset: str
    model: str
    variant: str
    stack: str
    source_type: str
    source_path: str
    step: int
    bbox_mAP: float
    bbox_mAP_50: float
    bbox_mAP_75: float
    bbox_mAP_s: float
    bbox_mAP_m: float
    bbox_mAP_l: float


LOCAL_RUNS = [
    {
        "model": "TinyViM_B",
        "variant": "Original TinyViM backbone",
        "stack": "MMDet3 + CUDA 12.8",
        "path": Path("artifacts/runs/visdrone_tinyvim_b_mmdet3_20260420_1408/eval_metrics.json"),
    },
    {
        "model": "HybridMamba-Base_B",
        "variant": "Low-frequency Mamba only",
        "stack": "MMDet3 + CUDA 12.8",
        "path": Path("artifacts/runs/visdrone_hybridmamba_base_b_mmdet3_20260420_1635/eval_metrics.json"),
    },
]

REMOTE_LOG = {
    "model": "HybridMambaDet_B",
    "variant": "Low-frequency Mamba + detail branch",
    "stack": "MMDet3 + CUDA 12.8 (FatMachine)",
    "path": Path("artifacts/tmp_validation/remote_visdrone_hybridmambadet_launcher.log"),
}

METRIC_KEYS = ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build authoritative VisDrone stage results from local eval files and remote logs.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tables"))
    return parser.parse_args()


def load_eval_metrics(path: Path, meta: dict[str, object]) -> RunRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunRecord(
        dataset="VisDrone2019-DET val",
        model=str(meta["model"]),
        variant=str(meta["variant"]),
        stack=str(meta["stack"]),
        source_type="eval_metrics.json",
        source_path=str(path),
        step=int(payload["step"]),
        bbox_mAP=float(payload["bbox_mAP"]),
        bbox_mAP_50=float(payload["bbox_mAP_50"]),
        bbox_mAP_75=float(payload["bbox_mAP_75"]),
        bbox_mAP_s=float(payload["bbox_mAP_s"]),
        bbox_mAP_m=float(payload["bbox_mAP_m"]),
        bbox_mAP_l=float(payload["bbox_mAP_l"]),
    )


def parse_remote_log(path: Path) -> RunRecord:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(
        r"Epoch\(val\)\s+\[(\d+)\]\[548/548\].*?coco/bbox_mAP:\s*([0-9.]+)\s+"
        r"coco/bbox_mAP_50:\s*([0-9.]+)\s+"
        r"coco/bbox_mAP_75:\s*([0-9.]+)\s+"
        r"coco/bbox_mAP_s:\s*([0-9.]+)\s+"
        r"coco/bbox_mAP_m:\s*([0-9.]+)\s+"
        r"coco/bbox_mAP_l:\s*([0-9.]+)",
        text,
    )
    if not matches:
        raise RuntimeError(f"Could not locate final validation metrics in {path}")
    step, ap, ap50, ap75, aps, apm, apl = matches[-1]
    return RunRecord(
        dataset="VisDrone2019-DET val",
        model=REMOTE_LOG["model"],
        variant=REMOTE_LOG["variant"],
        stack=REMOTE_LOG["stack"],
        source_type="remote_launcher.log",
        source_path=str(path),
        step=int(step),
        bbox_mAP=float(ap),
        bbox_mAP_50=float(ap50),
        bbox_mAP_75=float(ap75),
        bbox_mAP_s=float(aps),
        bbox_mAP_m=float(apm),
        bbox_mAP_l=float(apl),
    )


def render_markdown(records: list[RunRecord]) -> str:
    headers = ["Model", "Variant", "AP", "AP50", "AP75", "APs", "APm", "APl", "Source"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    record.model,
                    record.variant,
                    f"{record.bbox_mAP:.3f}",
                    f"{record.bbox_mAP_50:.3f}",
                    f"{record.bbox_mAP_75:.3f}",
                    f"{record.bbox_mAP_s:.3f}",
                    f"{record.bbox_mAP_m:.3f}",
                    f"{record.bbox_mAP_l:.3f}",
                    record.source_type,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[RunRecord] = []
    for item in LOCAL_RUNS:
        records.append(load_eval_metrics((repo_root / item["path"]).resolve(), item))
    records.append(parse_remote_log((repo_root / REMOTE_LOG["path"]).resolve()))

    records.sort(key=lambda record: ["TinyViM_B", "HybridMamba-Base_B", "HybridMambaDet_B"].index(record.model))

    csv_path = output_dir / "visdrone_stage_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    json_path = output_dir / "visdrone_stage_results.json"
    json_path.write_text(
        json.dumps(
            {
                "dataset": "VisDrone2019-DET val",
                "note": "Stage results under identical RetinaNet+FPN 1x protocol. HybridMambaDet is not yet the best variant and still needs tuning.",
                "records": [asdict(record) for record in records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md_path = output_dir / "visdrone_stage_results.md"
    md_path.write_text(render_markdown(records), encoding="utf-8")

    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "json": str(json_path),
                "markdown": str(md_path),
                "models": [record.model for record in records],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
