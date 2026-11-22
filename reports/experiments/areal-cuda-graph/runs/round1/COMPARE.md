# S3 (sync, CUDA graph, 10 steps, first skipped): before vs round 1

native-graph and rlkernel-before come from the earlier full run (same config, same cluster); rlkernel-round1 is S3-sync-graph in this directory. S3-profile* steps include torch.profiler post-processing on rank 0 and are attribution data only, not timing evidence.

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native-graph (before run) | sync | graph | FULL_DECODE_ONLY, capture 7s | 10 (1) | 728 | 1,440 | 10.28 | 38.47 | 18.08 | 72.67 | 3,102.5 | 8,470 |
| rlkernel-before | sync | graph | FULL_DECODE_ONLY, capture 25s | 10 (1) | 849 | 1,745 | 12.91 | 48.48 | 25.82 | 93.26 | 2,423.5 | 8,057 |
| rlkernel-round1 | sync | graph | FULL_DECODE_ONLY, capture 19s | 10 (1) | 662 | 1,069 | 13.24 | 13.24 | 6.59 | 39.05 | 2,383.2 | 3,513 |

## round 1 vs before (RL-Kernel arm)

| Metric | rlkernel-before | rlkernel-round1 | rlkernel-round1 / rlkernel-before | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 12.91 | 13.24 | 1.03x (2.6% slower) | [1.00x, 1.06x] |
| Rollout throughput (tok/GPU/s) | 2,423.46 | 2,383.24 | 0.98x (-1.7%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 8,056.90 | 3,512.60 | 0.44x (-56.4%) | - |
| Recompute logp time (s/step) | 25.82 | 6.59 | 0.26x (74.5% faster) | [0.24x, 0.27x] |
| Actor train time (s/step) | 48.48 | 13.24 | 0.27x (72.7% faster) | [0.26x, 0.29x] |
| Weight update time (s/step) | 2.62 | 2.60 | 0.99x (0.7% faster) | [0.98x, 1.01x] |
| End-to-end step time (s/step) | 93.26 | 39.05 | 0.42x (58.1% faster) | [0.40x, 0.44x] |

## round 1 vs native

| Metric | native-graph (before run) | rlkernel-round1 | rlkernel-round1 / native-graph (before run) | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 10.28 | 13.24 | 1.29x (28.9% slower) | [1.22x, 1.34x] |
| Rollout throughput (tok/GPU/s) | 3,102.55 | 2,383.24 | 0.77x (-23.2%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 8,470.50 | 3,512.60 | 0.41x (-58.5%) | - |
| Recompute logp time (s/step) | 18.08 | 6.59 | 0.36x (63.6% faster) | [0.36x, 0.37x] |
| Actor train time (s/step) | 38.47 | 13.24 | 0.34x (65.6% faster) | [0.34x, 0.35x] |
| Weight update time (s/step) | 2.68 | 2.60 | 0.97x (3.0% faster) | [0.95x, 0.99x] |
| End-to-end step time (s/step) | 72.67 | 39.05 | 0.54x (46.3% faster) | [0.53x, 0.55x] |
