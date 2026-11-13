# H100 re-run for #495 at head `fe2768f2` (2026-11-12/13)

4 x NVIDIA H100 80GB (driver 580.173.02), torch 2.13.0+cu130, vLLM 0.30.0, `KERNEL_ALIGN_FORCE_SM90=1`, `TORCH_CUDA_ARCH_LIST=9.0`. Two nodes with 4 H100 each, no 8-GPU node.

| check | result |
|---|---|
| logp backward, chunked, ordered merge, evidence, tolerance, op accuracy, loss step, comparison, gradient invariance, gtest (2 GPUs) | 455 passed, 18 skipped |

## What was run
The logp test files of the PR in the current layout (`tests/ops/logprob`, `tests/backends/cuda` logp files, `tests/benchmarks` logp evidence, `tests/contracts/test_tolerance_contract.py`, `tests/ops/test_op_accuracy.py`, `tests/e2e/test_rl_kernel_loss_step.py`, `tests/validation/...`) on 2 H100 visible, SM90 kernels built (`KERNEL_ALIGN_FORCE_SM90=1`). The 18 skips are Ascend kernels, MUSA, one CUDA-dispatch guard and one case that needs 4 GPUs. Timings not re-measured.

Raw logs are private; paths and host names are replaced (see `redaction.json`).
