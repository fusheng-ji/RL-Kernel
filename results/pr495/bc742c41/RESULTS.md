# #495 at `bc742c41` (full suite)

- GPU: 4 x NVIDIA H100 per node (95830 MiB each), driver 580.173.02, CUDA 13.0 (toolkit 13.x), torch 2.13.0+cu130, triton 3.7.1, python 3.12.15, vLLM 0.30.0. Exact per-run values: `header.txt`. Topology (`nvidia-smi.txt`): the 4 GPUs of a node are pairwise connected by NVLink (NV6); no PCIe-only GPU pair.
- Build: `TORCH_CUDA_ARCH_LIST=9.0`, `MAX_JOBS=32`, `ninja` on `PATH`. Before each GPU run `rl_engine._C.__file__` was printed and is inside the worktree of the run (`c-file.txt` / `steps.txt`).
- `git status --porcelain --untracked-files=no` was empty before (`header.txt`) and after (`status-after.txt`, empty) every run. No file in a worktree was edited; no patch was applied to any run in this directory.
- The shared environment was not modified. `ninja` (and, for a diagnostic of task A only, a different `apache-tvm-ffi`) come from separate directories.
- Node: the nodes were exclusive to this work. On node 1 two runs (#495 on GPUs 0-1, #538 on GPUs 2-3) ran at the same time in separate one-pair steps; all other runs were alone on their node, and no foreign process was visible (`header.txt`).

## Commands (`steps.txt`)
Two builds of the same head, each followed by the same pytest command on 2 GPUs (`CUDA_VISIBLE_DEVICES=0,1`):
1. `TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace` (clean `build/` and `_C*.so` first), then `python -m pytest -p no:cacheprovider -q -rs <14 files, see steps.txt>`: `pytest-default.log`
2. the same with `KERNEL_ALIGN_FORCE_SM90=1` at build and run: `pytest-sm90.log`

## Results
### pytest-default.log

Result line: `380 passed, 34 skipped in 82.88s (0:01:22)`

Skip reasons (`-rs`):

- 15 x batch_invariant_logp_sm90 kernel not compiled (needs KERNEL_ALIGN_FORCE_SM90=1 on an SM90/Hopper device).
- 15 x batch_invariant_logp_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 2 x SM90 fused logp requires a Hopper GPU and the extension built with SM90 enabled.
- 1 x CUDA dispatch may select fused logp
- 1 x cuda-sm90 backend is unavailable: batch_invariant_logp_sm90 is not compiled into the extension. Rebuild with KERNEL_ALIGN_FORCE_SM90=1 on an SM90 (Hopper) device: 'pip install -e .'

### pytest-sm90.log

Result line: `398 passed, 16 skipped in 70.53s (0:01:10)`

Skip reasons (`-rs`):

- 15 x batch_invariant_logp_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 1 x CUDA dispatch may select fused logp


## Reading
- Default build: 380 passed, 0 failed, 34 skipped. The 18 SM90-only tests are among the skips (15 `batch_invariant_logp_sm90` kernel-not-compiled, 2 `SM90 fused logp requires ... SM90 enabled`, 1 `cuda-sm90 backend is unavailable`). The other 16 skips are Ascend tests and one dispatch-dependent test.
- `KERNEL_ALIGN_FORCE_SM90=1` build: 398 passed, 0 failed, 16 skipped: the 18 SM90 tests ran and passed (380 + 18 = 398); the remaining 16 skips are the same Ascend / dispatch skips.
- The largest-N memory test was not skipped by the 80 GB limit on this 94 GB card (no memory-related skip reason).
- The performance and peak-memory check (`b1906159`) was not repeated, as instructed.

## Draft "H100" paragraph
Run on 4 x H100 nodes (2 GPUs used; driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130) at head `bc742c41`, detached clean worktree, no patch: the full suite listed in the handoff gives 380 passed / 0 failed / 34 skipped with `TORCH_CUDA_ARCH_LIST=9.0`, and 398 passed / 0 failed / 16 skipped with `KERNEL_ALIGN_FORCE_SM90=1`, where the 18 SM90 tests run and pass; the remaining skips are Ascend-only tests and one dispatch-dependent test. The fused kernel is unchanged since `61d30a0c`, so this repeats the suite of `a320e65a` at the current head.
