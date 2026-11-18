# #538 at `bdbd31ee`

- GPU: 4 x NVIDIA H100 per node (95830 MiB each), driver 580.173.02, CUDA 13.0 (toolkit 13.x), torch 2.13.0+cu130, triton 3.7.1, python 3.12.15, vLLM 0.30.0. Exact per-run values: `header.txt`. Topology (`nvidia-smi.txt`): the 4 GPUs of a node are pairwise connected by NVLink (NV6); no PCIe-only GPU pair.
- Build: `TORCH_CUDA_ARCH_LIST=9.0`, `MAX_JOBS=32`, `ninja` on `PATH`. Before each GPU run `rl_engine._C.__file__` was printed and is inside the worktree of the run (`c-file.txt` / `steps.txt`).
- `git status --porcelain --untracked-files=no` was empty before (`header.txt`) and after (`status-after.txt`, empty) every run. No file in a worktree was edited; no patch was applied to any run in this directory.
- The shared environment was not modified. `ninja` (and, for a diagnostic of task A only, a different `apache-tvm-ffi`) come from separate directories.
- Node: the nodes were exclusive to this work. On node 1 two runs (#495 on GPUs 0-1, #538 on GPUs 2-3) ran at the same time in separate one-pair steps; all other runs were alone on their node, and no foreign process was visible (`header.txt`).

## Commands
`KERNEL_ALIGN_USE_FAST_MATH=0 python setup.py build_ext --inplace` (clean first), then `KERNEL_ALIGN_USE_FAST_MATH=0 CUDA_VISIBLE_DEVICES=0,1 python -m pytest -p no:cacheprovider -q -rs tests/models/qwenimage` (same command as for the previous head; the merge of #705 adds `tests/distributed/sp/`, which is not part of this run).

## Results
### pytest.log

Result line: `67 passed in 217.45s (0:03:37)`


## Reading
67 passed, 0 failed, 0 skipped, including `test_final_scale_then_shift_and_layernorm`. Only within-H100 invariance is claimed (BF16 uses `mma.sync`); no comparison with B200 bytes.

## Draft "H100" paragraph
4 x H100 node (2 GPUs used; driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130), head `bdbd31ee` (the head after the #705 merge), detached clean worktree, build with `KERNEL_ALIGN_USE_FAST_MATH=0`, no patch: `tests/models/qwenimage` 67 passed / 0 failed / 0 skipped, including `test_final_scale_then_shift_and_layernorm`. The claim is within-H100 invariance only.
