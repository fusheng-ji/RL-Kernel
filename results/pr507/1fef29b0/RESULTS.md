# #507 at `1fef29b0` (run and report only; the PR is not edited)

- GPU: 4 x NVIDIA H100 per node (95830 MiB each), driver 580.173.02, CUDA 13.0 (toolkit 13.x), torch 2.13.0+cu130, triton 3.7.1, python 3.12.15, vLLM 0.30.0. Exact per-run values: `header.txt`. Topology (`nvidia-smi.txt`): the 4 GPUs of a node are pairwise connected by NVLink (NV6); no PCIe-only GPU pair.
- Build: `TORCH_CUDA_ARCH_LIST=9.0`, `MAX_JOBS=32`, `ninja` on `PATH`. Before each GPU run `rl_engine._C.__file__` was printed and is inside the worktree of the run (`c-file.txt` / `steps.txt`).
- `git status --porcelain --untracked-files=no` was empty before (`header.txt`) and after (`status-after.txt`, empty) every run. No file in a worktree was edited; no patch was applied to any run in this directory.
- The shared environment was not modified. `ninja` (and, for a diagnostic of task A only, a different `apache-tvm-ffi`) come from separate directories.
- Node: the nodes were exclusive to this work. On node 1 two runs (#495 on GPUs 0-1, #538 on GPUs 2-3) ran at the same time in separate one-pair steps; all other runs were alone on their node, and no foreign process was visible (`header.txt`).

## Commands (`steps.txt`)
Build with `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 TORCH_CUDA_ARCH_LIST=9.0`, `TRITON_F32_DEFAULT=ieee`, `VLLM_BATCH_INVARIANT=0`, then the commands of the PR description's H100 section: the three unit files, the seven `check_` modules, and `python -m torch.distributed.run --nproc-per-node 4 tools/validation/models/qwen3_next/qwen3_next_tp_{gdn,attention}_check.py --checkpoint <Qwen3-Next-80B-A3B-Instruct, revision 9c7f2fbe> --output <dir>`.

## Results
### pytest-check.log

Result line: `66 passed, 15 warnings in 50.46s`

### pytest-unit.log

Result line: `43 passed in 10.11s`


Gates: `tp_gdn/rank-{0..3}.json` and `tp_attention/rank-{0..3}.json`: `status: passed` on all four ranks, `rl_kernel_commit` `1fef29b0`, `tracked_tree_dirty: false`; both exit codes 0.

The D=256 attention **prior art** at the #507 head is not run from this head (the runner lives in #551); it is covered by task A step 5 at `2c0cd1e1` (see the Qwen3-Next results branch, `results/step5-pr507-attention-prior-art/`).

## Draft "H100" paragraph (for the coordinator)
4 x H100 node (driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, vLLM 0.30.0), checkpoint revision `9c7f2fbe`, head `1fef29b0`, SM90 flags (`KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1`), detached clean worktree, no patch: the three unit files (43 passed) and the seven `check_` modules (66 passed) from the description pass with 0 failed and 0 skipped, and the TP4 GDN and attention gates pass on all four ranks at this head (earlier evidence was at `0fdbf80b`).
