# Phase 05: Visualization And Analysis

## Goal

Produce the visual evidence that explains why the final model helps aerial tiny-object detection.

## Inputs

- stable baseline and final checkpoints
- exported evaluation tables
- selected images from VisDrone and AI-TOD-v2

## Tasks

1. Generate qualitative detections for baseline and final model.
2. Extract feature or response visualizations around small-object regions.
3. Produce high-frequency response maps and edge-emphasis examples.
4. Build AP versus FLOPs and AP versus throughput trade-off plots.
5. Save figure-generating code or notebooks into `scripts/` or `paper/figures/`.

## Outputs

- figure set for method explanation
- figure set for quantitative trade-off
- appendix-ready qualitative grids

## Risks

- qualitative examples look noisy or cherry-picked
- figure generation is not reproducible

## Exit Criteria

- every selected figure can be regenerated from scripts and stored artifacts
- the final figure set supports the paper claim boundary

