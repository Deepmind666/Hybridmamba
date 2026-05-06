# Phase 02: Hybrid Backbone Implementation

## Goal

Stabilize `HybridMamba-Base` and `HybridMambaDet` on `VisDrone` and lock the first usable structure for the paper.

## Inputs

- completed Phase 01 baseline
- `HybridMamba_Base_B` and `HybridMambaDet_B` code
- VisDrone converted dataset

## Tasks

1. Verify `HybridMamba_Base_B` can load the TinyViM checkpoint with non-fatal missing keys only.
2. Run smoke for `retinanet_hybridmamba_base_b_fpn_1x_visdrone.py`.
3. Run smoke for `retinanet_hybridmambadet_b_fpn_1x_visdrone.py`.
4. Compare first-run memory, stability, and loss curve behavior against the TinyViM baseline.
5. Sweep the minimum structural knobs:
   - `freq_split`
   - `detail_stages`
   - `fusion_alpha`
6. Freeze a default structure for the main paper model.

## Outputs

- stable HybridMamba baseline
- stable HybridMambaDet model
- first architecture decision log

## Risks

- checkpoint initialization mismatch
- excessive memory growth from detail branch
- no AP or AP_S improvement over baseline

## Exit Criteria

- final model trains under the same detector setup
- at least one HybridMambaDet setting is worth carrying into Phase 03

