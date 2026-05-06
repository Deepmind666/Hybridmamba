# Phase 01: Foundation And Baseline Reproduction

## Goal

Turn the repository into a reproducible training workspace and obtain the first valid `TinyViM + RetinaNet` smoke and baseline on `VisDrone`.

## Inputs

- `code/tinyvim`
- local WSL2 Ubuntu 24.04
- `VisDrone2019-DET` raw files
- `tinyvim_b_300e.pth`

## Tasks

1. Finish local Python 3.10 environment bootstrap with `torch`, `mmcv-full`, `mmdet`, `timm`, and `einops`.
2. Install or build `selective_scan_cuda` in the same environment.
3. Download and place the TinyViM B checkpoint into `weights/tinyvim/`.
4. Place raw VisDrone train and val images plus txt labels into `data/visdrone/`.
5. Convert VisDrone annotations into COCO with `scripts/convert_visdrone_to_coco.py`.
6. Validate converted JSON with `scripts/summarize_coco.py`.
7. Run `scripts/preflight_detection.py` on `retinanet_tinyvim_b_fpn_1x_visdrone.py`.
8. Run a 10-iteration smoke with `scripts/smoke_detection.py`.
9. If smoke passes, launch the first formal baseline run.

## Outputs

- working WSL environment
- valid VisDrone COCO annotations
- baseline run manifest
- baseline log and eval json

## Risks

- `mmcv-full` wheel compatibility
- missing TinyViM checkpoint
- VisDrone path mismatch after conversion
- RTX 5090 / `sm_120` incompatibility with the original `torch 2.0.1+cu118` stack

## Exit Criteria

- detector train entry runs without import errors
- 10-iteration smoke completes and writes `RUN_MANIFEST.json`, `train.log`, and `eval_metrics.json`
- one formal baseline run is launched or completed

Current note:

- Repository-side Phase 01 preparation is largely complete.
- The remaining blocker is runtime compatibility on RTX 5090 under the original TinyViM reference stack.
