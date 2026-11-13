# H100 re-run for #536 at head `e5373328` (2026-11-12/13)

4 x NVIDIA H100 80GB (driver 580.173.02), torch 2.13.0+cu130, vLLM 0.30.0, `KERNEL_ALIGN_FORCE_SM90=1`, `TORCH_CUDA_ARCH_LIST=9.0`. Two nodes with 4 H100 each, no 8-GPU node.

| check | result |
|---|---|
| det_gemm, RL_KERNEL_AUTOTUNE=0 | 73 passed, 9 skipped |
| det_gemm, RL_KERNEL_AUTOTUNE=1 | 73 passed, 9 skipped |
| persistent autotuner + kernel registry | 40 passed |
| autotuning GPU + candidates | 462 passed, 29 skipped |
| autotuning benchmark unit | 18 passed |

## What was run
The correctness part of the PR's earlier Hopper run, at the current head (which adds the `triton.__version__` fix in `autotuning.py` and tests for it). Timings (the benchmark part) were not re-measured.

Raw logs are private; paths and host names are replaced (see `redaction.json`).
