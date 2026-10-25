# PR #491 H100 summary

head: `cc27cf5cb565857abc1d78ae77d1541872e820d1`  
merge base (before benchmarks): `5ba769f681b7fa765ea6b29f9939454a858390ba`

## pytest

| suite | passed | failed | skipped |
| --- | --- | --- | --- |
| all_gather (3 files) | 40 | 0 | 1 |
| all_reduce / reduce_scatter | 0 | 0 | 2 (needs 8 visible GPUs; not run on this 4×H100 node) |

## benchmarks

2- and 4-GPU `benchmark_deterministic_collectives.py` on head and merge base (back-to-back).  
8-GPU benchmark and 8-GPU cross-TP tests: **not run** (no 8-GPU H100 allocation).

First benchmark attempt failed (login-node `torchrun` / missing `libcudart`); log preserved as `benchmark_after_w2.log.first_attempt`. Retry succeeded (`after_w2.json`, `after_w4.json`, `before_w*.json`).

Topology: four H100-94GB GPUs on one node (NVLink within node; no 8-GPU NVSwitch run).

## 64 KiB single-block threshold

See pytest `test_deterministic_all_gather_sizes` in `pytest_all_gather.log` (H100 holds unless log shows otherwise).
