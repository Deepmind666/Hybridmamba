# Execution Roadmap

Detailed working plans now live under [docs/phases](/C:/mamba/docs/phases/README.md). This file remains the compact overview.

## Phase 1: Base Import And Reproducibility

Deliverables:

- project rules and skills
- `TinyViM` base integrated into this repo
- WSL environment setup script
- dataset conversion scripts
- `TinyViM + RetinaNet` smoke config for `VisDrone`

Exit criteria:

- custom op can be imported in WSL
- detector train entry runs 10 iterations without error
- converted `VisDrone` COCO annotations pass sanity checks

## Phase 2: HybridMambaDet Implementation

Deliverables:

- `HybridMamba-Base` backbone
- `HybridMambaDet` backbone
- `VisDrone` configs for baseline and ablations

Exit criteria:

- backbone forward and backward pass succeed
- stage-level detail insertion is configurable
- ablation configs are executable without manual edits

## Phase 3: Multi-Dataset Transfer

Deliverables:

- `AI-TOD-v2` dataset config and normalized annotations
- `DOTA-HBB` conversion pipeline
- cross-dataset evaluation configs

Exit criteria:

- all three datasets produce valid COCO/HBB style annotations
- baseline and final model configs are portable across datasets

## Phase 4: Ablations And Result Export

Deliverables:

- five ablation groups
- run manifest convention
- result aggregation scripts

Exit criteria:

- eval json files can be exported into paper tables
- each ablation has a fixed config path and run id scheme

## Phase 5: Visualization Assets

Deliverables:

- high-frequency response figures
- edge / texture qualitative plots
- AP vs throughput / FLOPs trade-off plots

Exit criteria:

- every figure can be regenerated from saved artifacts

## Phase 6: Paper

Deliverables:

- English BMVC-style LaTeX draft
- tables and figures imported from artifact exporters
- citation audit workflow

Exit criteria:

- manuscript compiles
- claims match available evidence
