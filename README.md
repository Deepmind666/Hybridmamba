# HybridMambaDet: Lightweight Mamba Backbones for Aerial Object Detection

This repository studies lightweight Vision Mamba backbones for aerial object detection. The current priority is to first reproduce the TinyViM reference setting, then evaluate whether Mamba-style backbones transfer reliably to VisDrone2019 detection.

## Current Scope

- Reference reproduction: TinyViM-B on ImageNet-1K classification.
- Downstream detection: VisDrone2019-DET with TinyViM-style FPN/two-stage detection.
- Paper-aligned detector: TinyViM backbone + FPN + Mask R-CNN detection branch.
- VisDrone protocol: bbox-only training and evaluation, because VisDrone2019-DET does not provide instance masks.
- Exploratory baselines: MobileMamba and HybridMamba variants under the same detector and dataset protocol.

## Current Findings

- ImageNet-1K reproduction is the gating experiment. It verifies that the TinyViM implementation, pretrained teacher, data layout, and training recipe are aligned before downstream claims are made.
- Earlier VisDrone RetinaNet runs are treated as engineering probes rather than paper-aligned evidence. They are useful for debugging data conversion, training stability, and metric extraction, but they should not be used as the main comparison against the TinyViM paper.
- The corrected VisDrone setting follows the TinyViM detection protocol more closely: TinyViM-B is used as the backbone, FPN supplies multi-scale features, and the Mask R-CNN two-stage detection branch reports bbox AP. The mask branch is disabled for VisDrone because mask annotations are unavailable.
- Preliminary RetinaNet probes showed TinyViM-B as the stronger transferred backbone than MobileMamba-B1 on VisDrone, but the decisive comparison must be rerun under the paper-aligned FPN/two-stage detector.

## Repository Layout

```text
code/           upstream codebase and model changes
external/       read-only reference repos
data/           datasets and converted annotations
weights/        pretrained checkpoints
artifacts/      runs, eval json, tables, plots
docs/           status, roadmap, environment, idea notes
paper/          LaTeX manuscript and tables
scripts/        data conversion, smoke tests, result export
.claude/        project rules and skills
.codex/         Codex execution rules
```

Directory ownership and naming conventions are documented in [docs/folder-map.md](docs/folder-map.md). Phase-level execution plans are split under `docs/`.
The GitHub-facing project title and short description are fixed in [docs/project-title.md](docs/project-title.md).

## Working Rules

- WSL2 Ubuntu 24.04 is the primary execution environment.
- One machine runs at most one heavy full training job at a time.
- Every batch experiment must start with a smoke task.
- Tables and paper claims must be generated from run artifacts, never edited by hand.

## Upstream References

- `code/tinyvim`: main mutable codebase
- `external/mobilemamba`: structure and efficiency reference
- `external/pkinet`: remote sensing detail modeling reference
- `external/aitod`: official AI-TOD dataset construction reference
- `external/visdrone-dataset`: official dataset resources
- `external/vmamba`: selective-scan kernel reference
- `AI-TOD-v2 benchmark`: verified online, local clone still pending because the public repo times out during shallow fetch

See [docs/status.md](docs/status.md) and [docs/roadmap.md](docs/roadmap.md) for the active execution state.
If you are working on RTX 5090 / Blackwell GPUs, also read [docs/blackwell-migration.md](docs/blackwell-migration.md).
