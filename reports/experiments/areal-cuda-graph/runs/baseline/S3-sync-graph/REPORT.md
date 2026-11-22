# AReaL train/rollout consistency (sync)

| arm | verdict | steps | compared tokens | mismatching | max abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| native | REPORT_ONLY | 10 | 1273280 | - | 2.05475 |
| rlkernel | PASS | 10 | 1255011 | 0 | 0 |

## native

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129848 | - | 2.05475 | 0.0115599 | - | - | 0 |
| 1 | 130409 | - | 0.809499 | 0.0113876 | - | - | 0 |
| 2 | 130263 | - | 1.68891 | 0.0105213 | - | - | 0 |
| 3 | 128907 | - | 1.31985 | 0.0115028 | - | - | 0 |
| 4 | 126693 | - | 1.43883 | 0.0116882 | - | - | 0 |
| 5 | 124121 | - | 0.64167 | 0.0106061 | - | - | 0 |
| 6 | 126587 | - | 0.973964 | 0.0101835 | - | - | 0 |
| 7 | 126819 | - | 1.14911 | 0.0111075 | - | - | 0 |
| 8 | 126250 | - | 1.17375 | 0.0102233 | - | - | 0 |
| 9 | 123383 | - | 1.36565 | 0.0106977 | - | - | 0 |

## rlkernel

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129577 | 0 | 0 | 0 | 129577 | 0 | 0 |
| 1 | 130182 | 0 | 0 | 0 | 130182 | 0 | 0 |
| 2 | 128716 | 0 | 0 | 0 | 128716 | 0 | 0 |
| 3 | 128387 | 0 | 0 | 0 | 128387 | 0 | 0 |
| 4 | 126255 | 0 | 0 | 0 | 126255 | 0 | 0 |
| 5 | 122760 | 0 | 0 | 0 | 122760 | 0 | 0 |
| 6 | 124115 | 0 | 0 | 0 | 124115 | 0 | 0 |
| 7 | 124635 | 0 | 0 | 0 | 124635 | 0 | 0 |
| 8 | 121679 | 0 | 0 | 0 | 121679 | 0 | 0 |
| 9 | 118705 | 0 | 0 | 0 | 118705 | 0 | 0 |

## Performance (steady state, first step skipped)

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | sync | graph | FULL_DECODE_ONLY, capture 7s | 10 (1) | 728 | 1,440 | 10.28 | 38.47 | 18.08 | 72.67 | 3,102.5 | 8,470 |
| rlkernel | sync | graph | FULL_DECODE_ONLY, capture 25s | 10 (1) | 849 | 1,745 | 12.91 | 48.48 | 25.82 | 93.26 | 2,423.5 | 8,057 |

| Metric | native | rlkernel | rlkernel / native | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 10.28 | 12.91 | 1.26x (25.6% slower) | [1.22x, 1.29x] |
| Rollout throughput (tok/GPU/s) | 3,102.55 | 2,423.46 | 0.78x (-21.9%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 8,470.50 | 8,056.90 | 0.95x (-4.9%) | - |
| Recompute logp time (s/step) | 18.08 | 25.82 | 1.43x (42.8% slower) | [1.35x, 1.49x] |
| Actor train time (s/step) | 38.47 | 48.48 | 1.26x (26.0% slower) | [1.17x, 1.35x] |
| Weight update time (s/step) | 2.68 | 2.62 | 0.98x (2.3% faster) | [0.96x, 0.99x] |
| End-to-end step time (s/step) | 72.67 | 93.26 | 1.28x (28.3% slower) | [1.22x, 1.35x] |
