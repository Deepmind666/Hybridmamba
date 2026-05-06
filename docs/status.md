# Project Status

Last updated: 2026-04-21

## Objective

Turn the Hybrid Mamba idea into a paper-backed detection project with reproducible code, converted datasets, controlled experiments, and a BMVC-style English manuscript.

## Current Snapshot

- Repository root is established at `C:\mamba`.
- Upstream mutable base is imported at `code/tinyvim`.
- Reference repos are present locally:
  - `external/mobilemamba`
  - `external/pkinet`
  - `external/aitod`
  - `external/aitod-v2`
  - `external/visdrone-dataset`
- Core project rules, launch guards, and status tooling are now part of this repo.
- Implemented local additions remain in place:
  - `HybridMambaDet_B` and `HybridMamba_Base_B` backbones
  - `VisDrone`, `AI-TOD-v2`, and `DOTA-HBB` dataset configs
  - dataset conversion, smoke, result export, and guarded launch scripts
- Assets confirmed on disk:
  - `weights/tinyvim/tinyvim_b_300e.pth`
  - `data/visdrone/train/` and `data/visdrone/val/`
  - `data/converted/visdrone/annotations/train_coco.json`
  - `data/converted/visdrone/annotations/val_coco.json`
  - `data/aitodv2/_raw/annotations/`
- The strict TinyViM reference stack is still not usable on local `RTX 5090` hardware:
  - `torch 2.0.1+cu118` cannot execute on `sm_120`
  - the original `MMDet2` path therefore remains a reference-only baseline environment
- The migrated `MMDet3 + cu128` path is the active executable path:
  - `torch 2.7.1+cu128`
  - `mmengine 0.10.7`
  - `mmcv 2.2.0`
  - `mmdet 3.3.0`
  - `selective_scan_cuda_oflex` builds and imports successfully
- Migrated-stack smoke status is still valid:
  - `TinyViM_B + RetinaNet` smoke passed
  - `HybridMambaDet_B + RetinaNet` smoke passed

## Artifact-Backed Results

- The completed stage comparison under identical `RetinaNet + FPN` and `1x` settings is unchanged from `2026-04-20`:
  - `TinyViM_B`: `AP=0.007`, `AP50=0.019`, `AP75=0.004`, `AP-S=0.001`, `AP-M=0.010`, `AP-L=0.039`
  - `HybridMamba-Base_B`: `AP=0.015`, `AP50=0.035`, `AP75=0.011`, `AP-S=0.002`, `AP-M=0.024`, `AP-L=0.060`
  - `HybridMambaDet_B`: `AP=0.012`, `AP50=0.029`, `AP75=0.008`, `AP-S=0.002`, `AP-M=0.020`, `AP-L=0.054`
- These numbers are treated as stack-validation evidence, not final paper evidence.
- Polished stage artifacts remain available:
  - `artifacts/tables/visdrone_stage_results.csv`
  - `artifacts/figures/visdrone_stage_table_publication.png`
  - `artifacts/figures/visdrone_stage_metrics_publication.png`
  - `artifacts/figures/visdrone_stage_gain_publication.png`

## Active Experiment State

- The formal TinyViM reproduction target on `2026-04-21` has moved to the stronger `VisDrone V2` data and test protocol with a longer early-stop schedule:
  - config: `code/tinyvim/detection/configs_v3/retinanet_tinyvim_b_fpn_300e_visdrone_es_bs1_cpuassign.py`
- Latest local formal attempt:
  - run dir: `artifacts/runs/visdrone_tinyvim_b_mmdet3_300e_es_bs1_cpuassign_mem30_20260421_1126/`
  - created at: `2026-04-21 11:26:17`
  - last observed progress: `Epoch 1`, `iter 5200/6471`
  - latest observed loss: `0.6970`
  - latest observed ETA basis from log: `12 days, 2:14:30`
  - validation metrics: none yet
- This run stopped after the `iter 5200` log line with no Python traceback written to the run log.
- Earlier local attempts on the same day also stopped before first validation:
  - `artifacts/runs/visdrone_tinyvim_b_mmdet3_300e_es_bs1_cpuassign_mem30_20260421_0912/`
  - `artifacts/runs/visdrone_tinyvim_b_mmdet3_300e_es_bs1_cpuassign_mem30_20260421_0944/`
- Therefore, there are no new paper-usable validation metrics after the `2026-04-20` stage comparison.

## Execution Risk

- Local host policy changed materially on `2026-04-21`:
  - `scripts/check_local_host_stability.ps1` currently returns `blocked`
  - the recommended target is now `FatMachine`, not the local host
- The block is backed by current host-health evidence:
  - repeated unexpected shutdown records
  - recent `WHEA-Logger` hardware errors
  - elevated GPU activity during the check window
- Project consequence:
  - local machine is no longer trustworthy for paper-critical formal runs
  - local work should be limited to guarded smoke or preflight only after the checker returns `smoke_only` or `eligible`
  - the next formal reproduction must be routed to `FatMachine`

## Phase Status

- Phase 1: completed on the migrated `MMDet3 + cu128` stack
- Phase 2: in progress
- Phase 3: not started
- Phase 4: not started
- Phase 5: in progress
- Phase 6: not started

## Immediate Next Actions

1. Launch the next formal `TinyViM_B` reproduction on `FatMachine` through `scripts/start_recommended_training_blackwell.ps1 -Mode formal`.
2. Keep the local host on checker plus smoke duty only until the host-health guard stops returning `blocked`.
3. Once the first remote validation epoch lands, refresh `artifacts/tables/current_runs.*` and this status file immediately.
4. If the stronger TinyViM baseline still looks pathologically low, run `RetinaNet-R50-FPN` sanity under the same stronger protocol before touching `HybridMambaDet_B` again.
5. Keep `AI-TOD-v2` and `DOTA-HBB` as the next stage only after the stronger `VisDrone` baseline is credible.

## Hard Boundaries

- No detector innovation in this paper.
- No rotated detection pipeline in the mainline.
- `DOTA` is supplemental `HBB` evidence only.
- All paper tables must come from exported run artifacts.
