# Accelerated ImageNet Training Plan

This plan keeps the model, loss, batch size, and optimizer recipe unchanged. It accelerates data feeding only.

## Why This Is Needed

The current runs are bottlenecked by data access, not by GPU memory. Local TinyViM showed WSL `p9_client_rpc` waits while reading from `/mnt/c`, and Fat MobileMamba shows very low instantaneous GPU utilization while batch logs advance slowly. This points to Windows-mounted filesystem and small-file ImageNet loading as the main bottleneck.

## Prepared Data Targets

The scripts default to Linux ext4 locations inside each WSL environment. Override the target parameters if a different mount or disk is preferred.

## Prepared Scripts

Prepare local ext4 data:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_imagenet_ext4_local.ps1 -Background
```

Start TinyViM-B against local ext4 ImageNet:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_imagenet1k_tinyvim_local_ext4.ps1
```

Prepare Fat ext4 + LMDB data:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_imagenet_lmdb_fat.ps1
```

Start MobileMamba-B1 against Fat LMDB ImageNet:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_imagenet1k_mobilemamba_fat_lmdb.ps1
```

## Expected Runtime Impact

- Moving local TinyViM data from `/mnt/c` to WSL ext4 should reduce loader stalls and may improve throughput by about 1.5x-3x when I/O is the bottleneck.
- MobileMamba LMDB should reduce small-file overhead and may improve throughput by about 2x-5x after conversion.
- First-time copy/conversion is expensive: local ext4 sync may take 1-4 hours; Fat LMDB preparation may take 3-8 hours.

These are not official-checkpoint evaluation shortcuts. They are still training runs, only with a faster data backend.
