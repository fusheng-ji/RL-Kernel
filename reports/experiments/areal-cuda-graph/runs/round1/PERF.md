# AReaL x RL-Kernel performance and consistency: rlkg-r1-1122

## Consistency

| stage | arm | verdict | steps | compared tokens | mismatching | same-version tokens | same-version mismatching | max abs diff |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S3-profile | native-graph | REPORT_ONLY | 4 | 515643 | - | - | - | 1.00193 |
| S3-profile | rlkernel-graph | PASS | 4 | 517322 | 0 | - | - | 0 |
| S3-profile-base | rlkernel-graph | PASS | 4 | 515689 | 0 | - | - | 0 |
| S3-sync-graph | rlkernel-graph | PASS | 10 | 1264805 | 0 | - | - | 0 |

## Per arm (steady state, first 1 step(s) skipped)

### S3-profile

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native-graph | sync | graph | FULL_DECODE_ONLY, capture 7s | 4 (1) | 722 | 2,667 | 10.43 | 422.85 | 174.51 | 613.65 | 3,096.0 | 9,795 |
| rlkernel-graph | sync | graph | FULL_DECODE_ONLY, capture 19s | 4 (1) | 775 | 1,141 | 12.97 | 57.52 | 25.26 | 101.60 | 2,494.9 | 116 |

### S3-profile-base

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlkernel-graph | sync | graph | FULL_DECODE_ONLY, capture 25s | 4 (1) | 834 | 2,216 | 13.27 | 305.49 | 106.29 | 430.90 | 2,424.1 | 7,745 |

### S3-sync-graph

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlkernel-graph | sync | graph | FULL_DECODE_ONLY, capture 19s | 10 (1) | 662 | 1,069 | 13.24 | 13.24 | 6.59 | 39.05 | 2,383.2 | 3,513 |

## Comparisons

### sync: RL-Kernel cost with CUDA graphs on both arms

| Metric | native-graph | rlkernel-graph | rlkernel-graph / native-graph | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 10.43 | 13.24 | 1.27x (27.0% slower) | [1.15x, 1.40x] |
| Rollout throughput (tok/GPU/s) | 3,096.05 | 2,383.24 | 0.77x (-23.0%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 9,794.95 | 3,512.60 | 0.36x (-64.1%) | - |
| Recompute logp time (s/step) | 174.51 | 6.59 | 0.04x (96.2% faster) | [0.01x, 0.36x] |
| Actor train time (s/step) | 422.85 | 13.24 | 0.03x (96.9% faster) | [0.01x, 0.36x] |
| Weight update time (s/step) | 2.64 | 2.60 | 0.98x (1.6% faster) | [0.91x, 1.00x] |
| End-to-end step time (s/step) | 613.65 | 39.05 | 0.06x (93.6% faster) | [0.02x, 0.54x] |

## Notes

- Rollout throughput = generated tokens / timeperf/rollout / rollout GPUs.
- vLLM generation throughput = median of vLLM's 10 s log samples with running requests.
- Paired CI: bootstrap over steps paired by index (same seed and data order).
- Async rollout overlaps training, so compare async arms by end-to-end step time.
