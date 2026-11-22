# AReaL train/rollout consistency (sync)

| arm | verdict | steps | compared tokens | mismatching | max abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| rlkernel | PASS | 4 | 515689 | 0 | 0 |

## rlkernel

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129862 | 0 | 0 | 0 | 129862 | 0 | 0 |
| 1 | 129017 | 0 | 0 | 0 | 129017 | 0 | 0 |
| 2 | 128383 | 0 | 0 | 0 | 128383 | 0 | 0 |
| 3 | 128427 | 0 | 0 | 0 | 128427 | 0 | 0 |

## Performance (steady state, first step skipped)

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlkernel | sync | graph | FULL_DECODE_ONLY, capture 25s | 4 (1) | 834 | 2,216 | 13.27 | 305.49 | 106.29 | 430.90 | 2,424.1 | 7,745 |
