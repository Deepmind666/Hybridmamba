# Next Run Plan

## Active now
- Local: `local_aitodv2_tinyvim_stable_retry_mem14_20260501_002610`
  - Config: `retinanet_tinyvim_b_fpn_120e_aitodv2_stable.py`
  - Status at 2026-05-01 00:58: epoch 1 iter 800/11204, running.
  - First validation ETA: around 04:00-05:30 if the current local speed holds.
- Fat: `fat_aitodv2_hybridmambadet_stable_mem92_20260501_002325`
  - Config: `retinanet_hybridmambadet_b_fpn_120e_aitodv2_stable.py`
  - Status at 2026-05-01 00:59: epoch 1 iter 6300/11204, running without cooldown thrash in the latest tail.
  - First validation ETA: around 01:15-01:30 on 2026-05-01.
- Watcher: `C:\mamba\artifacts\queues\aitod_dual_watch_20260501_005834\watch.log`
  - Polls every 5 minutes.
  - If local exits before first validation with CUDA/Runtime failure, it launches a local-only mem12/260W/CPU0-3 retry.

## Current decisions
- `stage01` VisDrone ablation is negative evidence: original best AP 0.202, resumed best AP 0.195, latest observed AP 0.193.
- `fusion10` on Fat was stopped after epoch 25 and did not beat the main Fat baseline: best AP 0.203, latest AP 0.202.
- `fusion05` remains the strongest local-only Mamba ablation point: best AP 0.206, AP_S 0.127.

## Failed current-attempt runs
1. Local AI-TOD-v2 TinyViM baseline:
   `local_aitodv2_tinyvim_stable_20260430_212712`
   - Failed at epoch 1 iter 3950/11204 with `RuntimeError: CUDA error: unknown error`.
   - No validation metric was produced.

## Already running
1. Local AI-TOD-v2 TinyViM baseline:
   `retinanet_tinyvim_b_fpn_120e_aitodv2_stable.py`
   - Running as retry `local_aitodv2_tinyvim_stable_retry_mem14_20260501_002610`.
   - Full early-stop ETA is multi-day if it survives, because AI-TOD-v2 has 11204 iterations per epoch.
2. Fat AI-TOD-v2 HybridMambaDet final:
   `retinanet_hybridmambadet_b_fpn_120e_aitodv2_stable.py`
   - Running as `fat_aitodv2_hybridmambadet_stable_mem92_20260501_002325`.
   - Full early-stop ETA is roughly 14-24 hours if it keeps the current epoch speed and stops around epoch 20-30; this should be revised after the first validation.

## Still pending after the AI-TOD pair
1. Throughput + FLOPs export for:
   - TinyViM-B
   - HybridMamba-Base
   - HybridMambaDet
   - Fusion `alpha=0.5`
2. DOTA-HBB baseline + final, optional unless a second transfer dataset is needed.
3. Refresh paper figures/tables and update the HybridMambaDet write-up after AI-TOD results arrive.

## Ready launch entry points
- Local AI-TOD-v2 baseline direct launch:
  `powershell -NoProfile -ExecutionPolicy Bypass -File C:\mamba\scripts\start_local_training_blackwell_adaptive.ps1 -ConfigPath C:\mamba\code\tinyvim\detection\configs_v3\retinanet_tinyvim_b_fpn_120e_aitodv2_stable.py -RunId local_aitodv2_tinyvim_stable_YYYYMMDD_HHMMSS -GpuMemGb 16 -TorchNumThreads 1 -InteropThreads 1 -Niceness 15 -CpuCoreList 0-5 -GpuPowerLimitW 300 -GuardGpuUtilPct 80 -GuardCpuUtilPct 75 -GuardResumeUtilPct 65 -GuardTempC 74 -GuardMemoryPct 75 -GuardCooldownSec 90 -AllowBlockedHost`
- Fat AI-TOD-v2 final direct launch:
  `powershell -NoProfile -ExecutionPolicy Bypass -File C:\mamba\scripts\start_fatmachine_run_blackwell.ps1 -ConfigPath C:\mamba\code\tinyvim\detection\configs_v3\retinanet_hybridmambadet_b_fpn_120e_aitodv2_stable.py -RunId fat_aitodv2_hybridmambadet_stable_YYYYMMDD_HHMMSS -GpuMemGb 16 -TorchNumThreads 2 -InteropThreads 1 -AdaptiveGuard -GuardGpuUtilPct 80 -GuardCpuUtilPct 75 -GuardResumeUtilPct 65 -GuardTempC 74 -GuardMemoryPct 92 -GuardCheckIntervalSec 2 -GuardCooldownSec 60 -Background`

## Notes
- Use the current analysis bundle in `artifacts/analysis/mamba_current_20260430_0105/` as the source for paper figures and tables.
- Main rebuilt top-journal figure: `figure_mamba_topjournal_summary.png` with PDF/SVG variants.
- Throughput/FLOPs is still pending a detector-specific exporter script; do not mix it with classification `speed_gpu.py`.
- Every newly launched experiment must be reported to the user with an estimated completion time.
