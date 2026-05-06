# ImageNet-1K Quick Reproduction Plan

This note separates two different experiment modes:

1. **From-scratch training**: train TinyViM-B or MobileMamba-B1 on ImageNet-1K for many epochs. This is what the current long local/Fat jobs are doing. It is useful for checking whether our environment and training recipe can converge, but it is not the fastest way to reproduce the paper table.
2. **Official-checkpoint evaluation**: load the official pretrained checkpoint and evaluate on ImageNet-1K validation. This is the fast reproduction path and should be used first to verify that the dataset, codebase, checkpoint, and metrics match the paper.

## Expected Reference Numbers

- TinyViM-B official 300-epoch checkpoint: Top-1 around **81.2** on ImageNet-1K validation.
- MobileMamba-B1 official checkpoint: **Top-1 79.948**, **Top-5 94.924** on ImageNet-1K validation.

These numbers come from the local TinyViM paper/README and the official MobileMamba README checked in `external/mobilemamba`.

## Prepared Commands

Dry-run first. Dry-run does not start evaluation and does not interrupt the current long training jobs.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\eval_imagenet1k_tinyvim_b_local.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\eval_imagenet1k_mobilemamba_b1_fat.ps1 -DryRun
```

After the current long training jobs are stopped or finished, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\eval_imagenet1k_tinyvim_b_local.ps1
powershell -ExecutionPolicy Bypass -File scripts\eval_imagenet1k_mobilemamba_b1_fat.ps1
```

If we explicitly choose to evaluate while current training is still running, pass `-Force`. This will compete for GPU/CPU/memory and the ETA will be longer.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\eval_imagenet1k_tinyvim_b_local.ps1 -Force
powershell -ExecutionPolicy Bypass -File scripts\eval_imagenet1k_mobilemamba_b1_fat.ps1 -Force
```

## Outputs

- TinyViM-B local eval logs: `artifacts\eval\<run-id>\eval.log`
- MobileMamba-B1 Fat eval logs: `artifacts\eval\<run-id>\eval.log` under the remote project checkout.

## Runtime Estimate

- TinyViM-B official-checkpoint eval on local RTX 5090: about **20-45 minutes** when not sharing resources with training.
- MobileMamba-B1 official-checkpoint eval on Fat RTX 5090 D v2: about **15-45 minutes** when not sharing resources with training.

The current long jobs should not be interpreted as quick reproduction. They are 100-epoch from-scratch sanity runs on one GPU per machine, while the papers use long multi-GPU ImageNet training recipes.
