# Qwen3-Next distributed logprob on 4 x H100 (13 NCCL settings)

Date: 2026-11-14. One node, 4 x NVIDIA H100 (driver 580.173.02), official checkpoint
`Qwen/Qwen3-Next-80B-A3B-Instruct` revision `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`.

## Two commits

| part | directory | commit | tree state |
|---|---|---|---|
| logprob prior art (runner `benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py`) | `prior-art/` | `2c699e3f7cb30665b8e7dafe48508b5dc94b3ef4` (#551 head) | detached worktree, no patch, `tracked_tree_dirty: false` |
| TP4 parity runner (`tools/validation/models/qwen3_next/qwen3_next_logprob_parity.py` + `_compare.py`) | `parity/` | `30c0e5a7a432abcc4c03bed98c1c2ee914467f22` (#549 head) | detached worktree, no patch, `tracked_tree_dirty: false` |

The figure (`figure.png`) was drawn by the plot script at `2c699e3f` from both.
Each commit was built on its own (`python setup.py build_ext --inplace`,
`TORCH_CUDA_ARCH_LIST=9.0`, `KERNEL_ALIGN_FORCE_SM90=1`, `KERNEL_ALIGN_DET_GEMM_SM90=1`).

## Settings

13 TP4 launches in each part (`tp4-default`, `tp4-{ring,tree}-{ll,ll128,simple}-{c1,cdefault}`;
the parity part names them `default`, `ring-ll-c1`, ...), plus `tp1` for the prior art.
`ring` is `NCCL_ALGO=Ring`, `tree` is `NCCL_ALGO=allreduce:tree`; `c1` adds `NCCL_MAX_NCHANNELS=1`.
NCCL 2.29.7, torch 2.13.0+cu130, vLLM 0.30.0, Triton 3.7.1, transformers 5.17.0,
FlashInfer 0.6.18.post1, Megatron-core 0.16.0rc0.

## Deviations

* VIME is a source checkout installed in editable mode; its package metadata says `0.3.0`
  and `__version__` is not set (the handoff asked for 0.3.2; there is no PyPI package).
  The `vime_default` candidate loaded and was not `unavailable`.
* `miles_fused` and `verl_fused` are `unavailable` (not installed; optional in the handoff).
* The venv has an editable `rl_kernel` install that points at another worktree. Every run here
  sets `PYTHONPATH` to the worktree under test, and `rl_engine._C` was checked to come from it.
* No patch was applied to any tree in this directory.
