# AReaL train/rollout consistency (sync)

| arm | verdict | steps | compared tokens | mismatching | max abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| native | REPORT_ONLY | 4 | 515643 | - | 1.00193 |
| rlkernel | PASS | 4 | 517322 | 0 | 0 |

## native

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129488 | - | 0.860682 | 0.0114279 | - | - | 0 |
| 1 | 128215 | - | 0.862988 | 0.0115446 | - | - | 0 |
| 2 | 129217 | - | 0.697241 | 0.0102605 | - | - | 0 |
| 3 | 128723 | - | 1.00193 | 0.0115609 | - | - | 0 |

## rlkernel

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129470 | 0 | 0 | 0 | 129470 | 0 | 0 |
| 1 | 129488 | 0 | 0 | 0 | 129488 | 0 | 0 |
| 2 | 129407 | 0 | 0 | 0 | 129407 | 0 | 0 |
| 3 | 128957 | 0 | 0 | 0 | 128957 | 0 | 0 |

## Performance (steady state, first step skipped)

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | sync | graph | FULL_DECODE_ONLY, capture 7s | 4 (1) | 722 | 2,667 | 10.43 | 422.85 | 174.51 | 613.65 | 3,096.0 | 9,795 |
| rlkernel | sync | graph | FULL_DECODE_ONLY, capture 19s | 4 (1) | 775 | 1,141 | 12.97 | 57.52 | 25.26 | 101.60 | 2,494.9 | 116 |

| Metric | native | rlkernel | rlkernel / native | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 10.43 | 12.97 | 1.24x (24.4% slower) | [1.11x, 1.37x] |
| Rollout throughput (tok/GPU/s) | 3,096.05 | 2,494.95 | 0.81x (-19.4%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 9,794.95 | 115.50 | 0.01x (-98.8%) | - |
| Recompute logp time (s/step) | 174.51 | 25.26 | 0.14x (85.5% faster) | [0.13x, 0.38x] |
| Actor train time (s/step) | 422.85 | 57.52 | 0.14x (86.4% faster) | [0.12x, 0.38x] |
| Weight update time (s/step) | 2.64 | 2.61 | 0.99x (1.3% faster) | [0.98x, 0.99x] |
| End-to-end step time (s/step) | 613.65 | 101.60 | 0.17x (83.4% faster) | [0.13x, 0.55x] |
