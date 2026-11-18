# H100 evidence gaps, 2026-11-18: SUMMARY

Handoff branch `handoff/h100-evidence-gaps-20261114` @ `c6cd1ca0`. Only the six "Still open" rows of `STATUS-2026-11-18.md` were run; Priorities 1 and 2 were not rerun; #548 was skipped. Every head was re-checked with `gh pr view N --json headRefOid` on 2026-11-18 before building; all six equal the head named in the STATUS file (#538 `bdbd31ee`).

Common setup for all runs (details in each `RESULTS.md` and `header.txt`): 4 x H100 nodes, driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, python 3.12.15; one detached worktree per head, `git status --porcelain --untracked-files=no` empty before and after; `TORCH_CUDA_ARCH_LIST=9.0`; `rl_engine._C.__file__` checked to be inside the worktree. **No patch was applied in any run of this directory.** Failures are kept unedited and were not retried.

| PR | Head | What ran | passed / failed / skipped | Differs from expectation? |
|---|---|---|---|---|
| #507 | `1fef29b0` | 3 unit files, 7 `check_` modules, TP4 GDN and attention gates (4 GPUs, official checkpoint `9c7f2fbe`) | 43 / 0 / 0, 66 / 0 / 0; gates passed on 4 ranks each | no. The D=256 prior art is in the Qwen3-Next results (step 5, runner of #551). |
| #495 | `bc742c41` | full suite, 2 GPUs, default build and `KERNEL_ALIGN_FORCE_SM90=1` build | default 380 / 0 / 34; SM90 build 398 / 0 / 16 | no. The 18 SM90 tests skip in the default build and pass in the SM90 build. |
| #538 | `bdbd31ee` | `tests/models/qwenimage`, 2 GPUs, `KERNEL_ALIGN_USE_FAST_MATH=0` | 67 / 0 / 0 | no |
| #547 | `eb465fd8` | recursive test file, `MAX_ORDER=10` case, `tests/ops/logprob` + layout test, probe; `KERNEL_ALIGN_FORCE_SM90=1` | 89 / 0 / 0; 8 / 0 / 0; 640 / 0 / 17; probe verifies 48 traces | no |

## Held back
#536 (failures in `test_det_gemm.py` / `tests/ops`, also on `upstream/main`) and #491 (no 8-GPU run: the nodes have 4 GPUs) are not in this branch. Their evidence is on `handoff/h100-evidence-gaps-held-20261118`.

## Things to know
1. #538: the head moved to `bdbd31ee` as announced; the command is unchanged.
2. Two runs shared one node at the same time (#495 on two GPUs, #538 on the other two); this does not change test results but timings are not clean performance numbers.
3. Not done: #548 (skipped as instructed), Priorities 1 and 2.

Each `pr<N>/<head8>/RESULTS.md` has the header, commands, tables with skip reasons and a drafted "H100" paragraph. `nvidia-smi.txt` in each folder has `nvidia-smi --query-gpu=name,driver_version --format=csv` and `nvidia-smi topo -m`.
