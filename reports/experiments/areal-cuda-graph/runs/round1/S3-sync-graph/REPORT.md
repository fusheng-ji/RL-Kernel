# AReaL train/rollout consistency (sync)

| arm | verdict | steps | compared tokens | mismatching | max abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| rlkernel | PASS | 10 | 1264805 | 0 | 0 |

## rlkernel

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129814 | 0 | 0 | 0 | 129814 | 0 | 0 |
| 1 | 128898 | 0 | 0 | 0 | 128898 | 0 | 0 |
| 2 | 128686 | 0 | 0 | 0 | 128686 | 0 | 0 |
| 3 | 129355 | 0 | 0 | 0 | 129355 | 0 | 0 |
| 4 | 126145 | 0 | 0 | 0 | 126145 | 0 | 0 |
| 5 | 124699 | 0 | 0 | 0 | 124699 | 0 | 0 |
| 6 | 125198 | 0 | 0 | 0 | 125198 | 0 | 0 |
| 7 | 127211 | 0 | 0 | 0 | 127211 | 0 | 0 |
| 8 | 123888 | 0 | 0 | 0 | 123888 | 0 | 0 |
| 9 | 120911 | 0 | 0 | 0 | 120911 | 0 | 0 |

## Performance (steady state, first step skipped)

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlkernel | sync | graph | FULL_DECODE_ONLY, capture 19s | 10 (1) | 662 | 1,069 | 13.24 | 13.24 | 6.59 | 39.05 | 2,383.2 | 3,513 |
