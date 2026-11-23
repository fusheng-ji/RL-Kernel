# Round 1 extra runs: async (S4) and native sync with observe

Same configuration as the baseline run (Qwen3-8B, GSM8K, 2 x 4 H100, vLLM CUDA graphs, 10 steps, first skipped).

## Async, max off-policyness 1

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native-async (baseline run) | async | graph | FULL_DECODE_ONLY, capture 8s | 10 (1) | 778 | 1,392 | - | 37.73 | 18.01 | 61.81 | - | 4,321 |
| rlkernel-async-before | async | graph | FULL_DECODE_ONLY, capture 25s | 10 (1) | 901 | 1,734 | - | 44.81 | 24.36 | 75.04 | - | 8,227 |
| native-async (round 1 run) | async | graph | FULL_DECODE_ONLY, capture 7s | 10 (1) | 755 | 1,382 | - | 39.02 | 18.27 | 63.08 | - | 7,346 |
| rlkernel-async-round1 | async | graph | FULL_DECODE_ONLY, capture 20s | 10 (1) | 810 | 1,216 | - | 13.28 | 7.16 | 26.68 | - | 4,366 |

### round 1 RL-Kernel vs native, same session

| Metric | native-async (round 1 run) | rlkernel-async-round1 | rlkernel-async-round1 / native-async (round 1 run) | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | - | - | - | - |
| Rollout throughput (tok/GPU/s) | - | - | - | - |
| vLLM log generation throughput, coarse median (tok/s) | 7,345.80 | 4,365.65 | 0.59x (-40.6%) | - |
| Recompute logp time (s/step) | 18.27 | 7.16 | 0.39x (60.8% faster) | [0.37x, 0.42x] |
| Actor train time (s/step) | 39.02 | 13.28 | 0.34x (66.0% faster) | [0.34x, 0.35x] |
| Weight update time (s/step) | 2.64 | 2.76 | 1.04x (4.2% slower) | [1.03x, 1.06x] |
| End-to-end step time (s/step) | 63.08 | 26.68 | 0.42x (57.7% faster) | [0.42x, 0.43x] |

### round 1 RL-Kernel vs RL-Kernel before

| Metric | rlkernel-async-before | rlkernel-async-round1 | rlkernel-async-round1 / rlkernel-async-before | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | - | - | - | - |
| Rollout throughput (tok/GPU/s) | - | - | - | - |
| vLLM log generation throughput, coarse median (tok/s) | 8,226.80 | 4,365.65 | 0.53x (-46.9%) | - |
| Recompute logp time (s/step) | 24.36 | 7.16 | 0.29x (70.6% faster) | [0.28x, 0.31x] |
| Actor train time (s/step) | 44.81 | 13.28 | 0.30x (70.4% faster) | [0.28x, 0.31x] |
| Weight update time (s/step) | 2.64 | 2.76 | 1.04x (4.4% slower) | [1.03x, 1.06x] |
| End-to-end step time (s/step) | 75.04 | 26.68 | 0.36x (64.4% faster) | [0.35x, 0.37x] |

## Native sync: observe on vs off (observe overhead)

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native-sync (baseline run) | sync | graph | FULL_DECODE_ONLY, capture 7s | 10 (1) | 728 | 1,440 | 10.28 | 38.47 | 18.08 | 72.67 | 3,102.5 | 8,470 |
| native-sync-observe | sync | graph | FULL_DECODE_ONLY, capture 7s | 10 (1) | 748 | 1,566 | 10.18 | 38.56 | 18.07 | 72.55 | 3,104.4 | 7,604 |

| Metric | native-sync (baseline run) | native-sync-observe | native-sync-observe / native-sync (baseline run) | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 10.28 | 10.18 | 0.99x (0.9% faster) | [0.96x, 1.02x] |
| Rollout throughput (tok/GPU/s) | 3,102.55 | 3,104.44 | 1.00x (+0.1%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 8,470.50 | 7,604.20 | 0.90x (-10.2%) | - |
| Recompute logp time (s/step) | 18.08 | 18.07 | 1.00x (0.0% faster) | [0.99x, 1.01x] |
| Actor train time (s/step) | 38.47 | 38.56 | 1.00x (0.2% slower) | [0.99x, 1.02x] |
| Weight update time (s/step) | 2.68 | 2.63 | 0.98x (1.7% faster) | [0.97x, 1.00x] |
| End-to-end step time (s/step) | 72.67 | 72.55 | 1.00x (0.2% faster) | [0.99x, 1.01x] |
