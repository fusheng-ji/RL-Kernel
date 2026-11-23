# AReaL x RL-Kernel performance and consistency: rlkg-r2-1123

## Consistency

| stage | arm | verdict | steps | compared tokens | mismatching | same-version tokens | same-version mismatching | max abs diff |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S3-sync-observe | native-graph | REPORT_ONLY | 10 | 1264941 | 791080 | - | - | 1.275 |
| S4-async-graph | native-graph | REPORT_ONLY | 10 | 1278644 | 800880 | 129391 | 83318 | 1.76063 |
| S4-async-graph | rlkernel-graph | PASS | 10 | 1269976 | 499596 | 129234 | 0 | 1.74446 |

## Per arm (steady state, first 1 step(s) skipped)

### S3-sync-observe

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native-graph | sync | graph | FULL_DECODE_ONLY, capture 7s | 10 (1) | 748 | 1,566 | 10.18 | 38.56 | 18.07 | 72.55 | 3,104.4 | 7,604 |

### S4-async-graph

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native-graph | async | graph | FULL_DECODE_ONLY, capture 7s | 10 (1) | 755 | 1,382 | - | 39.02 | 18.27 | 63.08 | - | 7,346 |
| rlkernel-graph | async | graph | FULL_DECODE_ONLY, capture 20s | 10 (1) | 810 | 1,216 | - | 13.28 | 7.16 | 26.68 | - | 4,366 |

## Comparisons

### async: RL-Kernel cost with CUDA graphs on both arms

| Metric | native-graph | rlkernel-graph | rlkernel-graph / native-graph | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | - | - | - | - |
| Rollout throughput (tok/GPU/s) | - | - | - | - |
| vLLM log generation throughput, coarse median (tok/s) | 7,345.80 | 4,365.65 | 0.59x (-40.6%) | - |
| Recompute logp time (s/step) | 18.27 | 7.16 | 0.39x (60.8% faster) | [0.37x, 0.42x] |
| Actor train time (s/step) | 39.02 | 13.28 | 0.34x (66.0% faster) | [0.34x, 0.35x] |
| Weight update time (s/step) | 2.64 | 2.76 | 1.04x (4.2% slower) | [1.03x, 1.06x] |
| End-to-end step time (s/step) | 63.08 | 26.68 | 0.42x (57.7% faster) | [0.42x, 0.43x] |

## Notes

- Rollout throughput = generated tokens / timeperf/rollout / rollout GPUs.
- vLLM generation throughput = median of vLLM's 10 s log samples with running requests.
- Paired CI: bootstrap over steps paired by index (same seed and data order).
- Async rollout overlaps training, so compare async arms by end-to-end step time.
