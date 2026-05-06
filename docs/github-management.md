# GitHub Management Policy

This project is managed in the public repository:

```text
https://github.com/Deepmind666/Hybridmamba
```

## Tracked by Git

- Source code under `code/`.
- Reproducible experiment scripts under `scripts/`.
- Markdown/LaTeX project documentation under `docs/` and `paper/`.
- Lightweight experiment summaries under `artifacts/analysis/`.
- Publication figures and figure source summaries under `artifacts/figures/`.
- Small metadata files such as `data/README.md`, `weights/README.md`, and
  `references/reference_manifest.json`.

## Not Tracked by Git

- Raw datasets.
- Downloaded model weights.
- Training checkpoints.
- Full run directories and large training logs.
- Wheel caches, temporary archives, local render outputs, and scratch folders.
- Local agent state such as `.claude/` and `.codex/`.

## Current Experimental Framing

The ImageNet-1K TinyViM-B run is the original-paper reproduction task. It should
be reported with classification metrics such as Top-1 and Top-5 accuracy.

VisDrone2019 is a downstream detection-transfer task. The paper-aligned protocol
uses TinyViM as the backbone, FPN for multi-scale features, and the Mask R-CNN
two-stage detection branch. Since VisDrone2019-DET has bounding boxes but no
instance masks, this project reports bbox AP only for VisDrone.

Earlier RetinaNet-based VisDrone runs are retained as engineering probes for
data conversion, training stability, and plotting infrastructure. They should
not be treated as the main paper-aligned TinyViM detection result.
