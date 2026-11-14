Provider job at the stack top `2c699e3f7cb30665b8e7dafe48508b5dc94b3ef4` (#551; contains #543, #544, #549, #557), 4 x H100, 2026-11-14.
Detached worktree, `git status --short` empty before and after, no patch.
Steps: clean build of `rl_engine._C` (`build-summary.txt`), `tools/checks/ci_smoke.py`, the 13-file provider pytest run (149 passed, 0 skipped, 0 failed), then the 4-GPU model contract and EP4 real-model run (21 passed, 0 skipped, 0 failed).
`PYTHONPATH` was set to the worktree for every step after the build, because the venv has an editable `rl_kernel` install that points at another worktree; `ext.txt` shows `rl_engine._C` is loaded from the worktree under test (path redacted).
A first pass of the same steps without `PYTHONPATH` was discarded: its `ci_smoke.py` imported `_C` from the other worktree. Its pytest results were the same (149 and 21 passed).
Flags: `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 RL_KERNEL_REQUIRE_EXT=1 TRITON_F32_DEFAULT=ieee TORCH_CUDA_ARCH_LIST=9.0`.
