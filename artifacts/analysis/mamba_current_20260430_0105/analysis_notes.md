# Current Mamba Experiment Snapshot

Main Fat-machine comparable results use best validation checkpoints.

| Run | Status | Best epoch | Best AP | Latest AP | Best AP_S | Best AP_L |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TinyViM-B | complete | 12 | 0.204 | 0.197 | 0.124 | 0.347 |
| HybridMamba-Base | complete | 12 | 0.202 | 0.197 | 0.122 | 0.343 |
| HybridMambaDet | complete | 12 | 0.204 | 0.199 | 0.125 | 0.324 |
| Fusion alpha=1.0 | stopped | 12 | 0.203 | 0.202 | 0.123 | 0.330 |
| Fusion alpha=0.5 | complete | 13 | 0.206 | 0.201 | 0.127 | 0.327 |
| Stage shallow | stopped | 13 | 0.202 | 0.195 | 0.122 | 0.338 |

Key deltas against HybridMamba-Base (best checkpoints):
- HybridMambaDet: AP +0.002, AP_S +0.003, AP_L -0.019.
- Fusion alpha=1.0: AP +0.001, AP_S +0.001, AP_L -0.013.
- Fusion alpha=0.5 local: AP +0.004, AP_S +0.005, AP_L -0.016.
- Stage shallow: AP +0.000, AP_S +0.000, AP_L -0.005.

Interpretation: the VisDrone branch does not yet justify a stronger claim. The only consistent gain is the local alpha=0.5 point on AP/AP_S, but it still has no Fat-side confirmation and AP_L remains weaker. The resumed stage-shallow branch is now clearly negative evidence.

Next decisive run: AI-TOD-v2 baseline/final on local + Fat, then refresh figures and write-up.
