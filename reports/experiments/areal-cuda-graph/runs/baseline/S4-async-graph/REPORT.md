# AReaL train/rollout consistency (async)

| arm | verdict | steps | compared tokens | mismatching | max abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| native | REPORT_ONLY | 10 | 1274848 | 815585 | 2.73159 |
| rlkernel | PASS | 10 | 1273785 | 494321 | 1.64632 |

| arm | same-version tokens | same-version mismatching | mixed-seq same-version tokens | mixed mismatching |
| --- | ---: | ---: | ---: | ---: |
| native | 129764 | 84056 | 0 | 0 |
| rlkernel | 129431 | 0 | 0 | 0 |

## native

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129764 | 84056 | 1.23439 | 0.0116473 | 129764 | 84056 | 0 |
| 1 | 129072 | 83130 | 0.78419 | 0.0115276 | 0 | 0 | 1 |
| 2 | 129134 | 79251 | 0.76805 | 0.0105051 | 0 | 0 | 1 |
| 3 | 129400 | 84709 | 2.73159 | 0.0119163 | 0 | 0 | 1 |
| 4 | 127784 | 84580 | 0.579072 | 0.0121085 | 0 | 0 | 1 |
| 5 | 125279 | 79173 | 1.68089 | 0.0121104 | 0 | 0 | 1 |
| 6 | 126707 | 80169 | 0.861072 | 0.0115383 | 0 | 0 | 1 |
| 7 | 127659 | 80912 | 0.816842 | 0.0116773 | 0 | 0 | 1 |
| 8 | 126951 | 79917 | 1.03695 | 0.0127137 | 0 | 0 | 1 |
| 9 | 123098 | 79688 | 1.62306 | 0.0133235 | 0 | 0 | 1 |

## rlkernel

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129431 | 0 | 0 | 0 | 129431 | 0 | 0 |
| 1 | 129256 | 0 | 0 | 0 | 0 | 0 | 1 |
| 2 | 128410 | 57362 | 0.973316 | 0.0105851 | 0 | 0 | 1 |
| 3 | 128719 | 62926 | 1.64632 | 0.0119115 | 0 | 0 | 1 |
| 4 | 128217 | 65274 | 0.981293 | 0.0125229 | 0 | 0 | 1 |
| 5 | 125178 | 61296 | 0.975891 | 0.0125649 | 0 | 0 | 1 |
| 6 | 126551 | 60623 | 0.734444 | 0.0121746 | 0 | 0 | 1 |
| 7 | 127086 | 62506 | 0.961281 | 0.0127877 | 0 | 0 | 1 |
| 8 | 127923 | 62585 | 1.07512 | 0.0145563 | 0 | 0 | 1 |
| 9 | 123014 | 61749 | 1.5264 | 0.0148465 | 0 | 0 | 1 |

## Performance (steady state, first step skipped)

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | async | graph | FULL_DECODE_ONLY, capture 8s | 10 (1) | 778 | 1,392 | - | 37.73 | 18.01 | 61.81 | - | 4,321 |
| rlkernel | async | graph | FULL_DECODE_ONLY, capture 25s | 10 (1) | 901 | 1,734 | - | 44.81 | 24.36 | 75.04 | - | 8,227 |

| Metric | native | rlkernel | rlkernel / native | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | - | - | - | - |
| Rollout throughput (tok/GPU/s) | - | - | - | - |
| vLLM log generation throughput, coarse median (tok/s) | 4,320.60 | 8,226.80 | 1.90x (+90.4%) | - |
| Recompute logp time (s/step) | 18.01 | 24.36 | 1.35x (35.2% slower) | [1.32x, 1.39x] |
| Actor train time (s/step) | 37.73 | 44.81 | 1.19x (18.8% slower) | [1.15x, 1.22x] |
| Weight update time (s/step) | 2.63 | 2.64 | 1.00x (0.5% slower) | [0.99x, 1.03x] |
| End-to-end step time (s/step) | 61.81 | 75.04 | 1.21x (21.4% slower) | [1.19x, 1.24x] |
