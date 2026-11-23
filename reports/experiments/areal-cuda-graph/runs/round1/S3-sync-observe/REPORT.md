# AReaL train/rollout consistency (sync)

| arm | verdict | steps | compared tokens | mismatching | max abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| native | REPORT_ONLY | 10 | 1264941 | 791080 | 1.275 |

## native

- no issues

| step | compared | mismatching | max abs diff | avg abs diff | same-version | same mismatching | stale fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 128650 | 82169 | 0.739724 | 0.0113862 | 128650 | 82169 | 0 |
| 1 | 129160 | 81736 | 0.712848 | 0.0111762 | 129160 | 81736 | 0 |
| 2 | 130099 | 79470 | 1.0509 | 0.0104933 | 130099 | 79470 | 0 |
| 3 | 128911 | 83736 | 0.663499 | 0.011489 | 128911 | 83736 | 0 |
| 4 | 126741 | 83422 | 0.740898 | 0.0118681 | 126741 | 83422 | 0 |
| 5 | 123734 | 76889 | 0.716486 | 0.0103879 | 123734 | 76889 | 0 |
| 6 | 126150 | 76552 | 1.18811 | 0.010286 | 126150 | 76552 | 0 |
| 7 | 124946 | 76599 | 1.275 | 0.0106504 | 124946 | 76599 | 0 |
| 8 | 124890 | 75462 | 1.07739 | 0.0102637 | 124890 | 75462 | 0 |
| 9 | 121660 | 75045 | 0.57794 | 0.0105071 | 121660 | 75045 | 0 |

## Performance (steady state, first step skipped)

| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | sync | graph | FULL_DECODE_ONLY, capture 7s | 10 (1) | 748 | 1,566 | 10.18 | 38.56 | 18.07 | 72.55 | 3,104.4 | 7,604 |
