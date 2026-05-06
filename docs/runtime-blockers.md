# Runtime Blockers

Last updated: 2026-04-21

## Current Confirmed Facts

1. The repository has a working migrated detection path built around:
   - `torch 2.7.1+cu128`
   - `mmengine 0.10.7`
   - `mmcv 2.2.0`
   - `mmdet 3.3.0`
2. `selective_scan_cuda_oflex` builds and imports successfully on the migrated local stack.
3. The TinyViM checkpoint is present at:
   - `weights/tinyvim/tinyvim_b_300e.pth`
4. `VisDrone` train and val are present locally and converted into COCO JSON:
   - `data/converted/visdrone/annotations/train_coco.json`
   - `data/converted/visdrone/annotations/val_coco.json`
5. Stage-level `VisDrone` comparison artifacts were generated successfully on the migrated stack.
6. Guard scripts for host-health-aware launch routing now exist:
   - `scripts/check_local_host_stability.ps1`
   - `scripts/start_local_training_blackwell_guarded.ps1`
   - `scripts/start_recommended_training_blackwell.ps1`

## Current Hard Blockers

### 1. The original TinyViM reference stack is still unusable on local RTX 5090 hardware

The original reference environment remains:

- `torch 2.0.1 + cu118`
- `mmcv-full 1.7.2`
- `mmdet 2.28.2`

This stack imports, but CUDA execution fails on the local `RTX 5090` with:

- `RuntimeError: CUDA error: no kernel image is available for execution on the device`

The practical reason is unchanged: the reference stack does not support `sm_120`.

### 2. The local host is currently blocked for trustworthy formal training

On `2026-04-21`, `scripts/check_local_host_stability.ps1` returned `blocked`.

The blocking evidence includes:

- repeated `Kernel-Power 41` / `EventLog 6008` unexpected shutdown records
- recent `WHEA-Logger` hardware errors
- elevated GPU activity during the guard window

The current launcher policy therefore treats local formal runs as unsafe and routes formal work to `FatMachine`.

### 3. Recent local formal TinyViM reproductions are not stable enough to trust

The latest local formal attempt:

- `artifacts/runs/visdrone_tinyvim_b_mmdet3_300e_es_bs1_cpuassign_mem30_20260421_1126/`

reached:

- `Epoch 1`, `iter 5200/6471`

and then stopped without producing first-epoch validation metrics or a Python traceback.

Two earlier local attempts on the same day also stopped before first validation:

- `artifacts/runs/visdrone_tinyvim_b_mmdet3_300e_es_bs1_cpuassign_mem30_20260421_0912/`
- `artifacts/runs/visdrone_tinyvim_b_mmdet3_300e_es_bs1_cpuassign_mem30_20260421_0944/`

So the active blocker is no longer code importability. It is execution reliability on the local host.

## Consequence

The project is executable, but not all execution targets are trustworthy.

- software compatibility blocker: mitigated by the `MMDet3 + cu128` migration path
- local formal-training blocker: still active because the local host is unstable

This means the next credible formal baseline must be produced on `FatMachine`, not on the local workstation.

## What Is Still Valid

- repo structure
- migrated configs
- checkpoint path
- converted `VisDrone` annotations
- smoke tooling
- result export tooling
- stage-result figures and tables from `2026-04-20`

## Preferred Next Direction

### Remote-first for paper-critical work

Use:

- `scripts/start_recommended_training_blackwell.ps1 -Mode formal ...`

so the launcher routes the next formal reproduction to `FatMachine`.

### Local-only for guarded validation

Use the local host only when the guard allows it, and then only for:

- preflight
- smoke
- short validation

Do not launch local formal training again until the guard stops returning `blocked`.

## Higher-Level Engineering Risk

Even after the software migration succeeded, the project still has a hardware or platform stability risk on the local machine.

That risk must be treated as part of experiment integrity, not as an unrelated ops footnote, because interrupted first-epoch runs can otherwise be mistaken for model or config issues.
