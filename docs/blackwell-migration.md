# Blackwell Migration Path

This document tracks the controlled compatibility migration for RTX 5090 / `sm_120` hardware.

## Why this exists

The original TinyViM reference stack is:

- `torch 2.0.1 + cu118`
- `mmcv-full 1.7.2`
- `mmdet 2.28.2`

That stack is now preserved for experimental fidelity, but it does not execute on RTX 5090.

## Migration Goal

Create a parallel training path that can actually run on local Blackwell GPUs while preserving:

- the same dataset protocol
- the same detector choice
- the same backbone comparison logic

## Current Migration Target

- Python 3.10
- `torch 2.7.1 + cu128`
- `mmengine 0.10.7`
- `mmcv 2.2.0`
- `mmdet 3.3.0`

## Current Status

- environment installs successfully
- CUDA tensors run successfully on RTX 5090
- `mmcv 2.2.0` installs successfully
- `mmdet 3.3.0` imports after a local compatibility patch for the `mmcv 2.2.0` upper bound
- `selective_scan_cuda_oflex` builds successfully
- `TinyViM_B` and `HybridMambaDet_B` both pass `VisDrone` smoke
- the first formal `VisDrone` baseline run is active at:
  - `artifacts/runs/visdrone_tinyvim_b_mmdet3_20260420_1408/`
- latest extracted validation snapshot:
  - `bbox_mAP = 0.002`
  - `bbox_mAP_50 = 0.005`

## Parallel Assets

- environment script:
  - `scripts/setup_wsl_env_blackwell.sh`
- selective scan install script:
  - `scripts/install_selective_scan_blackwell.sh`
- mmdet compatibility patch:
  - `scripts/patch_mmdet_mmcv22.py`
- smoke script:
  - `scripts/smoke_detection_mmdet3.py`
- preflight script:
  - `scripts/preflight_detection_v3.py`
- training entry:
  - `scripts/run_train_mmdet3.py`
  - `scripts/run_local_training_blackwell.sh`
- config tree:
  - `code/tinyvim/detection/configs_v3/`

## Guardrails

- Do not delete the repaired `MMDet2` path.
- Treat this as a compatibility migration branch, not as evidence-equivalent baseline until it is validated.
- Do not merge paper claims across the old and new stacks without explicitly documenting the environment change.
