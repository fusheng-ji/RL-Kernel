# AReaL train/rollout consistency (sync)

| arm | verdict | steps | compared tokens | mismatching | max abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| rlkernel | PASS | 2 | 8192 | 0 | 0 |

## rlkernel

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4096 | 0 | 0 | 0 | 4096 | 0 | 0 |
| 1 | 4096 | 0 | 0 | 0 | 4096 | 0 | 0 |

## Performance (steady state, first step skipped)

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rlkernel | sync | graph | FULL_DECODE_ONLY, capture 25s | 2 (1) | 804 | 880 | 3.75 | 7.32 | 3.38 | 18.46 | 272.8 | 135 |
