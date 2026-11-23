# AReaL train/rollout consistency (async)

| arm | verdict | steps | compared tokens | mismatching | max abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| native | REPORT_ONLY | 10 | 1278644 | 800880 | 1.76063 |
| rlkernel | PASS | 10 | 1269976 | 499596 | 1.74446 |

| arm | same-version tokens | same-version mismatching | mixed-seq same-version tokens | mixed mismatching |
| --- | ---: | ---: | ---: | ---: |
| native | 129391 | 83318 | 0 | 0 |
| rlkernel | 129234 | 0 | 0 | 0 |

## native

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129391 | 83318 | 0.935305 | 0.0115941 | 129391 | 83318 | 0 |
| 1 | 130157 | 83442 | 0.727765 | 0.0113852 | 0 | 0 | 1 |
| 2 | 127926 | 78133 | 0.592088 | 0.0105495 | 0 | 0 | 1 |
| 3 | 128672 | 83604 | 1.13923 | 0.0117482 | 0 | 0 | 1 |
| 4 | 128413 | 84697 | 1.32941 | 0.0119767 | 0 | 0 | 1 |
| 5 | 125034 | 78914 | 0.984796 | 0.0127008 | 0 | 0 | 1 |
| 6 | 128104 | 79071 | 0.933998 | 0.0121393 | 0 | 0 | 1 |
| 7 | 127240 | 77539 | 0.729973 | 0.0120799 | 0 | 0 | 1 |
| 8 | 128689 | 77306 | 1.25572 | 0.0151615 | 0 | 0 | 1 |
| 9 | 125018 | 74856 | 1.76063 | 0.0158283 | 0 | 0 | 1 |

## rlkernel

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129234 | 0 | 0 | 0 | 129234 | 0 | 0 |
| 1 | 130040 | 0 | 0 | 0 | 0 | 0 | 1 |
| 2 | 129190 | 58364 | 0.68504 | 0.0107234 | 0 | 0 | 1 |
| 3 | 128428 | 64185 | 0.813663 | 0.0123166 | 0 | 0 | 1 |
| 4 | 127018 | 65586 | 0.680729 | 0.0126877 | 0 | 0 | 1 |
| 5 | 124680 | 62769 | 1.61945 | 0.0139122 | 0 | 0 | 1 |
| 6 | 128612 | 65228 | 0.954296 | 0.0134638 | 0 | 0 | 1 |
| 7 | 125747 | 60529 | 0.950844 | 0.0128994 | 0 | 0 | 1 |
| 8 | 125499 | 61175 | 1.17511 | 0.0153357 | 0 | 0 | 1 |
| 9 | 121528 | 61760 | 1.74446 | 0.016624 | 0 | 0 | 1 |

## Performance (steady state, first step skipped)

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | async | graph | FULL_DECODE_ONLY, capture 7s | 10 (1) | 755 | 1,382 | - | 39.02 | 18.27 | 63.08 | - | 7,346 |
| rlkernel | async | graph | FULL_DECODE_ONLY, capture 20s | 10 (1) | 810 | 1,216 | - | 13.28 | 7.16 | 26.68 | - | 4,366 |

| Metric | native | rlkernel | rlkernel / native | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | - | - | - | - |
| Rollout throughput (tok/GPU/s) | - | - | - | - |
| vLLM log generation throughput, coarse median (tok/s) | 7,345.80 | 4,365.65 | 0.59x (-40.6%) | - |
| Recompute logp time (s/step) | 18.27 | 7.16 | 0.39x (60.8% faster) | [0.37x, 0.42x] |
| Actor train time (s/step) | 39.02 | 13.28 | 0.34x (66.0% faster) | [0.34x, 0.35x] |
| Weight update time (s/step) | 2.64 | 2.76 | 1.04x (4.2% slower) | [1.03x, 1.06x] |
| End-to-end step time (s/step) | 63.08 | 26.68 | 0.42x (57.7% faster) | [0.42x, 0.43x] |
