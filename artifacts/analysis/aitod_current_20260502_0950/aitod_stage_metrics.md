# AI-TOD-v2 stage snapshot

| Run | Best AP | Latest AP | Best epoch | Latest epoch | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| TinyViM-B 1x | 16.6 | 16.6 | 11 | 12 | reference baseline |
| TinyViM retry | 7.2 | 5.1 | 3 | 4 | local retry baseline |
| HybridMamba Base | 5.2 | 5.2 | 3 | 3 | initial control |
| HybridMambaDet stable | 8.4 | 7.7 | 16 | 16 | best remote Fat checkpoint |
| HybridMambaDet fusion05 | 6.0 | 6.0 | 3 | 3 | current Fat snapshot |

## Live progress
- Local resumed base control: epoch 4, validation pending after restart.
- Fat resumed fusion05: epoch 4, validation pending after restart.

## Reading
- TinyViM-B 1x remains the strong benchmark at 16.6 AP.
- The current Mamba branch is still below the benchmark on AI-TOD-v2.
- Current live runs are still moving and have not crashed after the reboot.
