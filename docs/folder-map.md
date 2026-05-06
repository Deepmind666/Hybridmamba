# Folder Map

## Top-Level Ownership

| Path | Purpose | Editable | Notes |
| --- | --- | --- | --- |
| `code/` | primary mutable codebase | yes | all model and config changes live here |
| `external/` | read-only reference repos | mostly no | do not patch unless explicitly needed |
| `data/` | raw and converted datasets | yes | payload stays out of Git |
| `weights/` | checkpoints and initialization weights | yes | keep subfolders model-specific |
| `artifacts/` | run outputs, summaries, plots | yes | every formal run should be reproducible |
| `docs/` | planning, status, environment, structure | yes | operational source of truth |
| `paper/` | manuscript, figures, table exports | yes | only artifact-backed content |
| `scripts/` | conversion, smoke, preflight, export tools | yes | prefer deterministic utilities here |
| `.claude/` | project-local rules and skills | yes | collaboration layer |
| `.codex/` | Codex execution rules | yes | highest local execution priority |

## Canonical Subfolders

### `code/`

- `tinyvim/`: main mutable upstream tree

### `external/`

- `mobilemamba/`
- `pkinet/`
- `aitod/`
- `visdrone-dataset/`

### `data/`

- `visdrone/`: raw VisDrone files
- `aitodv2/`: raw AI-TOD-v2 files
- `dota/`: raw DOTA files
- `converted/visdrone/`
- `converted/aitodv2/`
- `converted/dota_hbb/`

### `weights/`

- `tinyvim/`
- `hybridmamba/`
- `detectors/`

### `artifacts/`

- `runs/`
- `tables/`
- `figures/`
- `tmp_validation/`: disposable validation scratch

## Naming Rules

- Configs: `retinanet_<backbone>_<dataset>.py`
- Run ids: `<dataset>_<model>_<schedule>_<seed>_<timestamp>`
- Exported tables: `<dataset>_<group>_<date>.csv` or `.tex`
- Figures: `<topic>_<dataset>_<date>.png`

## Cleanup Rules

- Keep `tmp_validation/` disposable.
- Remove accidental host-path directories immediately.
- Do not mix raw datasets and converted annotations in the same folder.
- Do not store manual result edits in `paper/` or `artifacts/`.

