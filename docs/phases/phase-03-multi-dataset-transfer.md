# Phase 03: Multi-Dataset Transfer

## Goal

Carry the locked backbone setup from `VisDrone` into `AI-TOD-v2` and `DOTA-HBB` without changing detector logic.

## Inputs

- completed Phase 02 model decision
- AI-TOD-v2 annotations and images
- DOTA raw images and labels

## Tasks

1. Normalize AI-TOD-v2 JSON with `scripts/convert_aitod_to_coco.py`.
2. Validate AI-TOD-v2 JSON structure and category continuity.
3. Convert DOTA to HBB patches with `scripts/convert_dota_hbb.py`.
4. Validate the generated patch annotations and patch image folders.
5. Run preflight on all AI-TOD-v2 and DOTA-HBB configs.
6. Smoke the TinyViM baseline on AI-TOD-v2.
7. Smoke the final HybridMambaDet model on AI-TOD-v2.
8. Smoke the TinyViM baseline on DOTA-HBB.
9. Smoke the final HybridMambaDet model on DOTA-HBB.

## Outputs

- normalized AI-TOD-v2 dataset
- DOTA-HBB patch dataset
- portable baseline and final configs across all three datasets

## Risks

- category mismatch in AI-TOD-v2
- patch explosion and storage growth on DOTA
- dataset-specific instability that breaks aligned comparison

## Exit Criteria

- all three datasets are preflight-clean except for optional missing weights
- baseline and final configs can run smoke on all datasets

