# ImageNet-1K Next Step

## Local

- Launcher: `scripts/start_imagenet1k_tinyvim_local.ps1`
- Default: `TinyViM_B`, `100` epochs, batch size `64`, base lr `0.004`
- Data root: `C:\mamba\data\imagenet`
- Teacher checkpoint: `C:\mamba\weights\tinyvim\regnety_160-a5fe301d.pth`
- Output: `C:\mamba\artifacts\runs\<run_id>\TinyViM_B\<timestamp>\`
- Required ImageNet layout: `train/<class>/*.JPEG` and `val/<class>/*.JPEG`; TinyViM can also fall back to `train.tar` and `val.tar`.

## Fat

- Launcher: `scripts/start_imagenet1k_mobilemamba_fat.ps1`
- Default: `MobileMamba_B1`, `100` epochs, batch size `256`
- Config: `configs/mobilemamba/mobilemamba_b1.py`
- Data root: `C:\Users\sshuser\data\imagenet` by default
- Output: `C:\Users\sshuser\codex_runs\hybrid-mamba\artifacts\runs\<run_id>\...`
- Required ImageNet layout: `train/<class>/*` and `val/<class>/*`; the launcher overrides MobileMamba from `ImageFolderLMDB` to `DefaultCLS`.

## Preflight

- Script: `scripts/preflight_imagenet1k_next.ps1`
- Asset/download script: `scripts/download_imagenet1k_assets.ps1`
- HF materializer: `scripts/materialize_imagenet1k_hf.py`
- Current check result on 2026-05-05 23:24:
  - Local RegNetY teacher is present.
  - Local ImageNet-1K root exists at `C:\mamba\data\imagenet`, but has no `train/val` data and no `train.tar/val.tar`.
  - Fat ImageNet-1K root exists at `C:\Users\sshuser\data\imagenet`, but has no `train/val` data.
  - Kaggle CLI is installed on both machines.
  - Hugging Face/Kaggle credentials are not configured on either machine.
  - Fat has the ImageNet helper scripts and TinyViM teacher synced under `C:\Users\sshuser\codex_runs\hybrid-mamba`.
  - Fat Windows Python has `kaggle`, `huggingface_hub`, `datasets`, `pyarrow`, `Pillow`, and `tqdm` installed for data download/materialization.

## Dataset Source Handling

- TinyViM teacher is downloaded from `https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth`.
- Full ImageNet-1K is gated. The automated path supports:
  - Hugging Face `ILSVRC/imagenet-1k` after `hf auth login` and access acceptance, then `scripts/download_imagenet1k_assets.ps1 -Source HuggingFace -MaterializeFromHf`.
  - Kaggle `imagenet-object-localization-challenge` after placing `kaggle.json` under `C:\Users\admin\.kaggle\kaggle.json`, then `scripts/download_imagenet1k_assets.ps1 -Source Kaggle`.
- Fat equivalent:
  - `powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\sshuser\codex_runs\hybrid-mamba\scripts\download_imagenet1k_assets.ps1 -Source HuggingFace -DatasetRoot C:\Users\sshuser\data\imagenet -DownloadRoot C:\Users\sshuser\data\downloads\imagenet1k -MaterializeFromHf`
  - or place `kaggle.json` under `C:\Users\sshuser\.kaggle\kaggle.json` and use `-Source Kaggle`.
- The preferred training layout for both baselines is `train/<class>/*` and `val/<class>/*`.

## Plotting

- Script: `scripts/plot_imagenet1k_publication.py`
- Style: white background, thin gray grid, AERIS-like blue/green palette
- Export: `PNG`, `PDF`, `SVG` plus CSV/Markdown summary
- Validated against:
  - TinyViM official `log.txt`
  - MobileMamba official `mobilemamba_b1.txt`

## Palette

- TinyViM-B: `#2D83BD`
- MobileMamba-B1: `#36A657`
- Reference / neutral: `#5A5A5A`
- Highlight: `#D15B9A`
