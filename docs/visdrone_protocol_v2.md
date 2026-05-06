# VisDrone Protocol V2

Last updated: 2026-04-20

## Why V2 Exists

The first `VisDrone` runs were useful for stack validation, but they are not strong enough to serve as paper evidence.

The main issues in the first round were:

- the schedule was only `12` epochs
- the resize policy stayed close to generic COCO defaults
- the detector kept a tight prediction cap (`max_per_img=100`)
- no sanity detector baseline was run under the migrated Blackwell stack

For aerial tiny-object detection, these choices are too weak.

## V2 Principles

This protocol keeps the paper boundary disciplined:

- detector family stays `RetinaNet + FPN`
- no loss redesign
- no assignment redesign
- no rotated detection
- no extra data beyond official full datasets

It only adjusts protocol-level knobs that are legitimate for rigorous reproduction:

- training length
- image scale
- optimizer schedule
- prediction candidate cap
- method-specific lightweight fusion knobs

## V2 Default Changes

### Shared detector protocol

- train / test image scale: `1600 x 960`
- schedule: `24` epochs
- warmup iterations: `1000`
- decay milestones: `16`, `22`
- detector test config:
  - `nms_pre = 2000`
  - `score_thr = 0.01`
  - `max_per_img = 300`

### Sanity detector

Use `RetinaNet-R50-FPN` as a protocol sanity check before trusting backbone conclusions.

Config:

- `code/tinyvim/detection/configs_v3/retinanet_r50_fpn_2x_visdrone_sanity.py`

Purpose:

- verify that the migrated `MMDet3 + cu128` stack can deliver a non-pathological VisDrone result under a stronger protocol
- separate pipeline issues from backbone issues

### TinyViM reproduction

Use `TinyViM_B` under the same `V2` detector protocol as the first official backbone reproduction target.

Config:

- `code/tinyvim/detection/configs_v3/retinanet_tinyvim_b_fpn_2x_visdrone_v2.py`

Purpose:

- establish a stronger official-style tiny hybrid Mamba baseline before returning to `HybridMambaDet`

## Allowed Tuning Knobs

These are allowed in the next stage:

- epochs
- resize scale
- warmup length
- learning rate
- optimizer family
- weight decay
- batch size
- gradient clipping
- `nms_pre`
- `score_thr`
- `max_per_img`
- `freq_split`
- `detail_stages`
- `fusion_alpha`

## Forbidden Changes

These remain forbidden:

- changing detector type
- changing neck or head family
- changing assignment algorithm
- changing loss family for the main paper comparison
- adding extra unofficial data
- using ensemble tricks
- manually altering any reported metric

## Execution Order

1. `RetinaNet-R50-FPN` sanity on `VisDrone`
2. `TinyViM_B` on `VisDrone V2`
3. only after `TinyViM_B` is credible, return to `HybridMambaDet`
