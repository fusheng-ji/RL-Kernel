# AReaL x RL-Kernel performance and consistency: rlkg-out-1122-0435

## Consistency

| stage | arm | verdict | steps | compared tokens | mismatching | same-version tokens | same-version mismatching | max abs diff |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S2-smoke-graph | rlkernel-graph | PASS | 2 | 8192 | 0 | - | - | 0 |
| S3-sync-graph | native-graph | REPORT_ONLY | 10 | 1273280 | - | - | - | 2.05475 |
| S3-sync-graph | rlkernel-graph | PASS | 10 | 1255011 | 0 | - | - | 0 |
| S4-async-graph | native-graph | REPORT_ONLY | 10 | 1274848 | 815585 | 129764 | 84056 | 2.73159 |
| S4-async-graph | rlkernel-graph | PASS | 10 | 1273785 | 494321 | 129431 | 0 | 1.64632 |
| S5-sync-eager | native-eager | REPORT_ONLY | 10 | 1274564 | - | - | - | 2.97716 |
| S5-sync-eager | rlkernel-eager | PASS | 10 | 1264929 | 0 | - | - | 0 |

## Per arm (steady state, first 1 step(s) skipped)

### S2-smoke-graph

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlkernel-graph | sync | graph | FULL_DECODE_ONLY, capture 25s | 2 (1) | 804 | 880 | 3.75 | 7.32 | 3.38 | 18.46 | 272.8 | 135 |

### S3-sync-graph

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native-graph | sync | graph | FULL_DECODE_ONLY, capture 7s | 10 (1) | 728 | 1,440 | 10.28 | 38.47 | 18.08 | 72.67 | 3,102.5 | 8,470 |
| rlkernel-graph | sync | graph | FULL_DECODE_ONLY, capture 25s | 10 (1) | 849 | 1,745 | 12.91 | 48.48 | 25.82 | 93.26 | 2,423.5 | 8,057 |

### S4-async-graph

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native-graph | async | graph | FULL_DECODE_ONLY, capture 8s | 10 (1) | 778 | 1,392 | - | 37.73 | 18.01 | 61.81 | - | 4,321 |
| rlkernel-graph | async | graph | FULL_DECODE_ONLY, capture 25s | 10 (1) | 901 | 1,734 | - | 44.81 | 24.36 | 75.04 | - | 8,227 |

### S5-sync-eager

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native-eager | sync | eager | eager (graphs disabled) | 10 (1) | 737 | 1,587 | 26.91 | 37.58 | 18.01 | 88.22 | 1,182.5 | 5,101 |
| rlkernel-eager | sync | eager | eager (graphs disabled) | 10 (1) | 982 | 2,615 | 95.25 | 47.64 | 25.63 | 174.84 | 331.1 | 1,395 |

## Comparisons

### sync: RL-Kernel cost with CUDA graphs on both arms

| Metric | native-graph | rlkernel-graph | rlkernel-graph / native-graph | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 10.28 | 12.91 | 1.26x (25.6% slower) | [1.22x, 1.29x] |
| Rollout throughput (tok/GPU/s) | 3,102.55 | 2,423.46 | 0.78x (-21.9%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 8,470.50 | 8,056.90 | 0.95x (-4.9%) | - |
| Recompute logp time (s/step) | 18.08 | 25.82 | 1.43x (42.8% slower) | [1.35x, 1.49x] |
| Actor train time (s/step) | 38.47 | 48.48 | 1.26x (26.0% slower) | [1.17x, 1.35x] |
| Weight update time (s/step) | 2.68 | 2.62 | 0.98x (2.3% faster) | [0.96x, 0.99x] |
| End-to-end step time (s/step) | 72.67 | 93.26 | 1.28x (28.3% slower) | [1.22x, 1.35x] |

### sync: CUDA graph gain on the RL-Kernel arm

| Metric | rlkernel-eager | rlkernel-graph | rlkernel-graph / rlkernel-eager | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 95.25 | 12.91 | 0.14x (86.4% faster) | [0.13x, 0.14x] |
| Rollout throughput (tok/GPU/s) | 331.08 | 2,423.46 | 7.32x (+632.0%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 1,394.60 | 8,056.90 | 5.78x (+477.7%) | - |
| Recompute logp time (s/step) | 25.63 | 25.82 | 1.01x (0.7% slower) | [0.98x, 1.02x] |
| Actor train time (s/step) | 47.64 | 48.48 | 1.02x (1.8% slower) | [0.99x, 1.05x] |
| Weight update time (s/step) | 2.66 | 2.62 | 0.98x (1.7% faster) | [0.96x, 1.01x] |
| End-to-end step time (s/step) | 174.84 | 93.26 | 0.53x (46.7% faster) | [0.52x, 0.55x] |

### sync: CUDA graph gain on the native arm

| Metric | native-eager | native-graph | native-graph / native-eager | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 26.91 | 10.28 | 0.38x (61.8% faster) | [0.37x, 0.40x] |
| Rollout throughput (tok/GPU/s) | 1,182.51 | 3,102.55 | 2.62x (+162.4%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 5,100.60 | 8,470.50 | 1.66x (+66.1%) | - |
| Recompute logp time (s/step) | 18.01 | 18.08 | 1.00x (0.4% slower) | [0.98x, 1.02x] |
| Actor train time (s/step) | 37.58 | 38.47 | 1.02x (2.3% slower) | [1.00x, 1.05x] |
| Weight update time (s/step) | 2.55 | 2.68 | 1.05x (5.2% slower) | [1.03x, 1.08x] |
| End-to-end step time (s/step) | 88.22 | 72.67 | 0.82x (17.6% faster) | [0.81x, 0.84x] |

### sync: Previous configuration: native eager vs RL-Kernel graph

| Metric | native-eager | rlkernel-graph | rlkernel-graph / native-eager | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 26.91 | 12.91 | 0.48x (52.0% faster) | [0.47x, 0.49x] |
| Rollout throughput (tok/GPU/s) | 1,182.51 | 2,423.46 | 2.05x (+104.9%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 5,100.60 | 8,056.90 | 1.58x (+58.0%) | - |
| Recompute logp time (s/step) | 18.01 | 25.82 | 1.43x (43.4% slower) | [1.36x, 1.51x] |
| Actor train time (s/step) | 37.58 | 48.48 | 1.29x (29.0% slower) | [1.21x, 1.37x] |
| Weight update time (s/step) | 2.55 | 2.62 | 1.03x (2.8% slower) | [1.01x, 1.04x] |
| End-to-end step time (s/step) | 88.22 | 93.26 | 1.06x (5.7% slower) | [1.01x, 1.10x] |

### sync: RL-Kernel cost, both arms eager (previous runs)

| Metric | native-eager | rlkernel-eager | rlkernel-eager / native-eager | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 26.91 | 95.25 | 3.54x (254.0% slower) | [3.51x, 3.57x] |
| Rollout throughput (tok/GPU/s) | 1,182.51 | 331.08 | 0.28x (-72.0%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 5,100.60 | 1,394.60 | 0.27x (-72.7%) | - |
| Recompute logp time (s/step) | 18.01 | 25.63 | 1.42x (42.4% slower) | [1.36x, 1.49x] |
| Actor train time (s/step) | 37.58 | 47.64 | 1.27x (26.8% slower) | [1.22x, 1.32x] |
| Weight update time (s/step) | 2.55 | 2.66 | 1.05x (4.5% slower) | [1.02x, 1.08x] |
| End-to-end step time (s/step) | 88.22 | 174.84 | 1.98x (98.2% slower) | [1.94x, 2.02x] |

### async: RL-Kernel cost with CUDA graphs on both arms

| Metric | native-graph | rlkernel-graph | rlkernel-graph / native-graph | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | - | - | - | - |
| Rollout throughput (tok/GPU/s) | - | - | - | - |
| vLLM log generation throughput, coarse median (tok/s) | 4,320.60 | 8,226.80 | 1.90x (+90.4%) | - |
| Recompute logp time (s/step) | 18.01 | 24.36 | 1.35x (35.2% slower) | [1.32x, 1.39x] |
| Actor train time (s/step) | 37.73 | 44.81 | 1.19x (18.8% slower) | [1.15x, 1.22x] |
| Weight update time (s/step) | 2.63 | 2.64 | 1.00x (0.5% slower) | [0.99x, 1.03x] |
| End-to-end step time (s/step) | 61.81 | 75.04 | 1.21x (21.4% slower) | [1.19x, 1.24x] |

## Notes

- Rollout throughput = generated tokens / timeperf/rollout / rollout GPUs.
- vLLM generation throughput = median of vLLM's 10 s log samples with running requests.
- Paired CI: bootstrap over steps paired by index (same seed and data order).
- Async rollout overlaps training, so compare async arms by end-to-end step time.
