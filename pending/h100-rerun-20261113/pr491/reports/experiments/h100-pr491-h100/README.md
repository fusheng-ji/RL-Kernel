# H100 re-run for #491 at head `8c03d134` (2026-11-12/13)

4 x NVIDIA H100 80GB (driver 580.173.02), torch 2.13.0+cu130, vLLM 0.30.0, `KERNEL_ALIGN_FORCE_SM90=1`, `TORCH_CUDA_ARCH_LIST=9.0`. Two nodes with 4 H100 each, no 8-GPU node.

| check | result |
|---|---|
| collective tests (2 and 4 GPUs visible) | 62 passed, 2 skipped (8 CUDA GPUs; 8 Ascend NPUs) |

## What was run
The PR's collective tests and `benchmarks/distributed/benchmark_deterministic_collectives.py` at world 2 and 4. "Before" is the merge base `76a127c2` with the PR's current benchmark script copied in; "after" is the PR head. No 8-GPU run (4-GPU nodes only).

all_gather median, deterministic path (us):

| bytes per rank | w2 before | w2 after | w4 before | w4 after | NCCL (after run, w4) |
|---|---|---|---|---|---|
| 1024 | 29.8 | 26.6 | 40.1 | 31.1 | 72.0 |
| 65536 | 272.1 | 36.4 | 640.4 | 41.6 | 57.8 |
| 262144 | 48.4 | 45.9 | 61.1 | 51.1 | 58.4 |
| 1048576 | 58.4 | 55.2 | 59.8 | 56.0 | 97.2 |
| 4194304 | 104.8 | 84.0 | 125.1 | 131.4 | 110.8 |

At 4 MiB per rank on 4 GPUs the new path is not faster (131.4 vs 125.1 us); the gain is at 64 KiB.

Raw logs are private; paths and host names are replaced (see `redaction.json`).
