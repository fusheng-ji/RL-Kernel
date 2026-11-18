# #536 at `4a83e4f8`, diff against `upstream/main` `49941801` on the same box

- GPU: 4 x NVIDIA H100 per node (95830 MiB each), driver 580.173.02, CUDA 13.0 (toolkit 13.x), torch 2.13.0+cu130, triton 3.7.1, python 3.12.15, vLLM 0.30.0. Exact per-run values: `header.txt`. Topology (`nvidia-smi.txt`): the 4 GPUs of a node are pairwise connected by NVLink (NV6); no PCIe-only GPU pair.
- Build: `TORCH_CUDA_ARCH_LIST=9.0`, `MAX_JOBS=32`, `ninja` on `PATH`. Before each GPU run `rl_engine._C.__file__` was printed and is inside the worktree of the run (`c-file.txt` / `steps.txt`).
- `git status --porcelain --untracked-files=no` was empty before (`header.txt`) and after (`status-after.txt`, empty) every run. No file in a worktree was edited; no patch was applied to any run in this directory.
- The shared environment was not modified. `ninja` (and, for a diagnostic of task A only, a different `apache-tvm-ffi`) come from separate directories.
- Node: the nodes were exclusive to this work. On node 1 two runs (#495 on GPUs 0-1, #538 on GPUs 2-3) ran at the same time in separate one-pair steps; all other runs were alone on their node, and no foreign process was visible (`header.txt`).

## Commands
Build with `TORCH_CUDA_ARCH_LIST=9.0` (no SM90 flag, as in the handoff), `RL_KERNEL_MODE=auto`, a private fresh `RL_KERNEL_AUTOTUNE_CACHE_DIR` (a new temp directory per command, so no cache dump exists), `RL_KERNEL_AUTOTUNE={0,1}` for the GEMM file and for `tests/ops`; the autotune suite once. `python -m pytest -p no:cacheprovider -q -rs ...`. All 4 GPUs of the node visible (the TP=4 case runs). The same build and the `det_gemm` / `tests/ops` commands were run on `upstream/main` right after (same node, same GPUs): `upstream-main-49941801/`. The case that exercises `e5373328` (skip tuning when the profiling copies do not fit) is in `tests/platforms/test_persistent_autotuner.py`, which is part of the autotune suite.

## Results, head `4a83e4f8`
### autotune-suite.log

Result line: `520 passed, 29 skipped in 671.63s (0:11:11)`

Skip reasons (`-rs`):

- 27 x Existing unsupported dtype
- 2 x Existing operator does not support this dtype

### det-gemm-autotune0.log

Result line: `18 failed, 55 passed, 9 skipped in 25.76s`

Skip reasons (`-rs`):

- 9 x leaf specialization raw-byte tests require ROCm gfx942

### det-gemm-autotune1.log

Result line: `18 failed, 55 passed, 9 skipped in 9.74s`

Skip reasons (`-rs`):

- 9 x leaf specialization raw-byte tests require ROCm gfx942

### ops-autotune0.log

Result line: `816 failed, 1541 passed, 401 skipped in 146.57s (0:02:26)`

Skip reasons (`-rs`):

- 72 x rmsnorm_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 70 x bitwise expf/logf sequence is only ported for ROCm
- 48 x Ascend NPU required
- 42 x current torch build is not ROCm platform
- 40 x CUDA FlashAttentionOp is unavailable: Neither external flash_attn nor _C.flash_attn_forward is available.
- 29 x ROCm-only acceptance check
- 27 x Existing unsupported dtype
- 19 x Fused linear log-prob SM90 kernel requires a Hopper (sm_90) GPU with the extension built KERNEL_ALIGN_FORCE_SM90=1.
- 15 x batch_invariant_logp_sm90 kernel not compiled (needs KERNEL_ALIGN_FORCE_SM90=1 on an SM90/Hopper device).
- 15 x batch_invariant_logp_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 9 x leaf specialization raw-byte tests require ROCm gfx942
- 7 x rope_apply_ascend is not compiled (requires an Ascend NPU build).
- 2 x Hopper with compiled RMSNorm and RoPE CUDA kernels is required
- 2 x Existing operator does not support this dtype
- 2 x SM90 fused logp requires a Hopper GPU and the extension built with SM90 enabled.
- 1 x Triton linear log-prob requires a MUSA device and Triton.
- 1 x CUDA dispatch may select fused logp

### ops-autotune1.log

Result line: `816 failed, 1541 passed, 401 skipped in 199.37s (0:03:19)`

Skip reasons (`-rs`):

- 72 x rmsnorm_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 70 x bitwise expf/logf sequence is only ported for ROCm
- 48 x Ascend NPU required
- 42 x current torch build is not ROCm platform
- 40 x CUDA FlashAttentionOp is unavailable: Neither external flash_attn nor _C.flash_attn_forward is available.
- 29 x ROCm-only acceptance check
- 27 x Existing unsupported dtype
- 19 x Fused linear log-prob SM90 kernel requires a Hopper (sm_90) GPU with the extension built KERNEL_ALIGN_FORCE_SM90=1.
- 15 x batch_invariant_logp_sm90 kernel not compiled (needs KERNEL_ALIGN_FORCE_SM90=1 on an SM90/Hopper device).
- 15 x batch_invariant_logp_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 9 x leaf specialization raw-byte tests require ROCm gfx942
- 7 x rope_apply_ascend is not compiled (requires an Ascend NPU build).
- 2 x Hopper with compiled RMSNorm and RoPE CUDA kernels is required
- 2 x Existing operator does not support this dtype
- 2 x SM90 fused logp requires a Hopper GPU and the extension built with SM90 enabled.
- 1 x Triton linear log-prob requires a MUSA device and Triton.
- 1 x CUDA dispatch may select fused logp


## Results, `upstream/main` `49941801`
### det-gemm-autotune0.log

Result line: `2 failed, 20 passed, 49 skipped in 8.32s`

Skip reasons (`-rs`):

- 40 x Triton is unavailable
- 9 x leaf specialization raw-byte tests require ROCm gfx942

### det-gemm-autotune1.log

Result line: `2 failed, 20 passed, 49 skipped in 8.04s`

Skip reasons (`-rs`):

- 40 x Triton is unavailable
- 9 x leaf specialization raw-byte tests require ROCm gfx942

### ops-autotune0.log

Result line: `332 failed, 1512 passed, 412 skipped in 78.23s (0:01:18)`

Skip reasons (`-rs`):

- 72 x rmsnorm_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 70 x bitwise expf/logf sequence is only ported for ROCm
- 48 x Ascend NPU required
- 42 x current torch build is not ROCm platform
- 40 x CUDA FlashAttentionOp is unavailable: Neither external flash_attn nor _C.flash_attn_forward is available.
- 40 x Triton is unavailable
- 29 x ROCm-only acceptance check
- 19 x Fused linear log-prob SM90 kernel requires a Hopper (sm_90) GPU with the extension built KERNEL_ALIGN_FORCE_SM90=1.
- 15 x batch_invariant_logp_sm90 kernel not compiled (needs KERNEL_ALIGN_FORCE_SM90=1 on an SM90/Hopper device).
- 15 x batch_invariant_logp_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 9 x leaf specialization raw-byte tests require ROCm gfx942
- 7 x rope_apply_ascend is not compiled (requires an Ascend NPU build).
- 2 x Hopper with compiled RMSNorm and RoPE CUDA kernels is required
- 2 x SM90 fused logp requires a Hopper GPU and the extension built with SM90 enabled.
- 1 x Triton linear log-prob requires a MUSA device and Triton.
- 1 x CUDA dispatch may select fused logp

### ops-autotune1.log

Result line: `332 failed, 1512 passed, 412 skipped in 64.61s (0:01:04)`

Skip reasons (`-rs`):

- 72 x rmsnorm_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 70 x bitwise expf/logf sequence is only ported for ROCm
- 48 x Ascend NPU required
- 42 x current torch build is not ROCm platform
- 40 x CUDA FlashAttentionOp is unavailable: Neither external flash_attn nor _C.flash_attn_forward is available.
- 40 x Triton is unavailable
- 29 x ROCm-only acceptance check
- 19 x Fused linear log-prob SM90 kernel requires a Hopper (sm_90) GPU with the extension built KERNEL_ALIGN_FORCE_SM90=1.
- 15 x batch_invariant_logp_sm90 kernel not compiled (needs KERNEL_ALIGN_FORCE_SM90=1 on an SM90/Hopper device).
- 15 x batch_invariant_logp_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 9 x leaf specialization raw-byte tests require ROCm gfx942
- 7 x rope_apply_ascend is not compiled (requires an Ascend NPU build).
- 2 x Hopper with compiled RMSNorm and RoPE CUDA kernels is required
- 2 x SM90 fused logp requires a Hopper GPU and the extension built with SM90 enabled.
- 1 x Triton linear log-prob requires a MUSA device and Triton.
- 1 x CUDA dispatch may select fused logp


## Reading (not edited, not retried)
- Autotune suite (`autotune-suite.log`, includes the `e5373328` tests): 520 passed, 0 failed, 29 skipped.
- `tests/ops/gemm/test_det_gemm.py` and `tests/ops` have failures on **both** the head and `upstream/main`: the first failing test is the same one on both, `test_target_shapes_invariance[shape3-cuda-deterministic_gemm]`, with `CUDA error: an illegal memory access was encountered`. A CUDA illegal-address error is sticky in the process, so every CUDA test that follows in the same pytest process fails with the same message; the failure counts therefore depend on how many tests follow the first error in each run, and are not counts of independent failures:

| Command | head `4a83e4f8` (passed / failed / skipped) | `upstream/main` `49941801` |
|---|---|---|
| `test_det_gemm.py`, `RL_KERNEL_AUTOTUNE=0` | 55 / 18 / 9 | 20 / 2 / 49 |
| `test_det_gemm.py`, `RL_KERNEL_AUTOTUNE=1` | 55 / 18 / 9 | 20 / 2 / 49 |
| `tests/ops`, `RL_KERNEL_AUTOTUNE=0` | 1541 / 816 / 401 | 1512 / 332 / 412 |
| `tests/ops`, `RL_KERNEL_AUTOTUNE=1` | 1541 / 816 / 401 | 1512 / 332 / 412 |

- The head's `test_det_gemm.py` has more tests than main's (82 vs 71 collected), and the skipped counts differ (9 vs 49). So the counts of the two columns are not comparable test by test; what is comparable is that the same test is the first to hit the illegal memory access on both, in a build without `KERNEL_ALIGN_FORCE_SM90`. This looks like a pre-existing H100 problem of the non-SM90-flag GEMM build, not something introduced by this PR; it was not investigated further, and whether the same test passes with `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1` was not run.
- `RL_KERNEL_AUTOTUNE=0` and `=1` gave identical counts.

## Draft "H100" paragraph
4 x H100 node (driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130), head `4a83e4f8`, build `TORCH_CUDA_ARCH_LIST=9.0`, detached clean worktree, private autotune cache directory, no patch. The autotune suite (persistent autotuner incl. the `e5373328` tests, kernel registry, autotuning GPU/candidates/benchmark tests) gives 520 passed / 0 failed / 29 skipped. `tests/ops/gemm/test_det_gemm.py` and `tests/ops` fail after a CUDA illegal memory access in `test_target_shapes_invariance[shape3-cuda-deterministic_gemm]`; the same test is the first to fail on `upstream/main` `49941801` on the same node and build, so the failures are not introduced by this PR (counts and caveat in the results; all failures are kept unedited).
