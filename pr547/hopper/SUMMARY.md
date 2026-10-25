# PR #547 H100 summary

head: `ff1ad6350050b3b11bec86364f5c1bfaecf23f78`  
build: `KERNEL_ALIGN_FORCE_SM90=1`, `TORCH_CUDA_ARCH_LIST=9.0`

| suite | passed | failed | skipped |
| --- | --- | --- | --- |
| `test_deterministic_logp_recursive.py` | 89 | 0 | 0 |
| recursive_orders dtype3 (`RL_KERNEL_LOGP_MAX_ORDER=10`) | 8 | 0 | 0 |
| `tests/ops/logprob` + layout | 640 | 0 | 17 |

Probe: `recursive-backends.json` (48 CUDA traces verified).

SM90-only cases run on H100 vs skip on B200; compare via diff vs merge-base counts on same box, not absolute B200 totals.
