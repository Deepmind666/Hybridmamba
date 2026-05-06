# Phase 04: Ablations And Results Aggregation

## Goal

Turn experiments into structured evidence that can be exported into paper-ready tables.

## Inputs

- stable baseline and final model
- multi-dataset configs
- run manifest and eval json conventions

## Tasks

1. Define the ablation grid for:
   - frequency decoupling necessity
   - detail branch structure
   - insertion stages
   - Mamba input type
   - lightweight gain versus heavier local alternatives
2. Launch one ablation group at a time.
3. Keep each run tied to a config path and manifest.
4. Export results with `scripts/export_results_table.py`.
5. Build intermediate summary CSV and Markdown tables in `artifacts/tables/`.
6. Mark which comparisons are paper-mainline and which stay appendix-only.

## Outputs

- artifact-backed comparison tables
- ablation summary tables
- keep/drop decision for each experiment group

## Risks

- too many runs without a clean naming discipline
- AP improves but AP_S does not
- detector variance overwhelms backbone signal

## Exit Criteria

- paper-mainline results have stable exported tables
- appendix-only experiments are clearly separated

