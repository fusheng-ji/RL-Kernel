# AReaL train/rollout consistency (sync)

| arm | verdict | steps | compared tokens | mismatching | max abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| native | REPORT_ONLY | 10 | 1274564 | - | 2.97716 |
| rlkernel | PASS | 10 | 1264929 | 0 | 0 |

## native

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129430 | - | 0.740398 | 0.0114098 | - | - | 0 |
| 1 | 129537 | - | 0.673521 | 0.0115289 | - | - | 0 |
| 2 | 128803 | - | 0.842348 | 0.010462 | - | - | 0 |
| 3 | 129240 | - | 2.97716 | 0.0116465 | - | - | 0 |
| 4 | 126952 | - | 1.33193 | 0.0117286 | - | - | 0 |
| 5 | 124333 | - | 0.625528 | 0.0106954 | - | - | 0 |
| 6 | 128426 | - | 1.28831 | 0.0105321 | - | - | 0 |
| 7 | 127540 | - | 0.806378 | 0.0108914 | - | - | 0 |
| 8 | 126732 | - | 0.813523 | 0.0104907 | - | - | 0 |
| 9 | 123571 | - | 0.841709 | 0.0106859 | - | - | 0 |

## rlkernel

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129595 | 0 | 0 | 0 | 129595 | 0 | 0 |
| 1 | 129137 | 0 | 0 | 0 | 129137 | 0 | 0 |
| 2 | 128967 | 0 | 0 | 0 | 128967 | 0 | 0 |
| 3 | 129520 | 0 | 0 | 0 | 129520 | 0 | 0 |
| 4 | 128062 | 0 | 0 | 0 | 128062 | 0 | 0 |
| 5 | 123977 | 0 | 0 | 0 | 123977 | 0 | 0 |
| 6 | 125460 | 0 | 0 | 0 | 125460 | 0 | 0 |
| 7 | 125608 | 0 | 0 | 0 | 125608 | 0 | 0 |
| 8 | 123343 | 0 | 0 | 0 | 123343 | 0 | 0 |
| 9 | 121260 | 0 | 0 | 0 | 121260 | 0 | 0 |

## Performance (steady state, first step skipped)

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | sync | eager | eager (graphs disabled) | 10 (1) | 737 | 1,587 | 26.91 | 37.58 | 18.01 | 88.22 | 1,182.5 | 5,101 |
| rlkernel | sync | eager | eager (graphs disabled) | 10 (1) | 982 | 2,615 | 95.25 | 47.64 | 25.63 | 174.84 | 331.1 | 1,395 |

| Metric | native | rlkernel | rlkernel / native | paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Rollout time (s/step) | 26.91 | 95.25 | 3.54x (254.0% slower) | [3.51x, 3.57x] |
| Rollout throughput (tok/GPU/s) | 1,182.51 | 331.08 | 0.28x (-72.0%) | - |
| vLLM log generation throughput, coarse median (tok/s) | 5,100.60 | 1,394.60 | 0.27x (-72.7%) | - |
| Recompute logp time (s/step) | 18.01 | 25.63 | 1.42x (42.4% slower) | [1.36x, 1.49x] |
| Actor train time (s/step) | 37.58 | 47.64 | 1.27x (26.8% slower) | [1.22x, 1.32x] |
| Weight update time (s/step) | 2.55 | 2.66 | 1.05x (4.5% slower) | [1.02x, 1.08x] |
| End-to-end step time (s/step) | 88.22 | 174.84 | 1.98x (98.2% slower) | [1.94x, 2.02x] |
