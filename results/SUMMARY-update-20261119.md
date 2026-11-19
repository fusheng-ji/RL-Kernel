# Qwen3-Next stack on H100: update of 2026-11-19 (rerun with the environment fix)

Appended after `SUMMARY.md` (2026-11-18). That file is unchanged; where it says "step 0 fails", "chains not run" or "7 of 10 candidates", this update supersedes it. Same hardware (4 x H100 per node, driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, vLLM 0.30.0, python 3.12.15), same SM90 flags, same detached clean worktrees (`git status --porcelain --untracked-files=no` empty before and after, see each `header.txt` / `status-after.txt`). No Hopper patch was applied (`hopper_recompute_no_ddp.patch` was not needed). Each run's `rl_engine._C` is inside its worktree or the launcher clone (the launcher builds it in `preflight`).

## Environment fix (not a code change)
`~/.local` supplied apache-tvm-ffi 0.1.16 over the 0.1.11 of the base environment. For the probes it was put first on `PYTHONPATH` as a separate directory (nothing was installed into the shared env). The Qwen3-Next launcher overwrites `PYTHONPATH` with its source clone, so for the chains the fix is `PYTHONNOUSERSITE=1` (the base environment then gives 0.1.11). Three more environment items were needed and are not code issues:
- `core.fileMode=false` through `GIT_CONFIG_*` variables (this filesystem reports mode changes, so the launcher's `source_mismatch` check fired at once; attempt 1 in `not-green-attempts/`);
- `CUDNN_HOME` and `LD_LIBRARY_PATH` pointing at the nccl and cudnn directories of the base environment (Transformer Engine could not find `libcudnn.so`; this is a library search path, not the user directory);
- `CUDA_DEVICE_MAX_CONNECTIONS=1` (Megatron asserts it for `l2`; first `l2` attempts failed on exactly that assertion).

## What ran
| Step | Result |
|---|---|
| 0: six `run_probe.sh` cases with 0.1.11 | **all 6 rc 0, no `__ffi_repr__` abort** (`step0-rerun-.../res/summary.txt`) |
| 0: real 80B smoke, `VLLM_ALLREDUCE_USE_FLASHINFER=0`, `2c699e3f`, TP4, `--non-strict-smoke` | **rc 0**, two prompts generated 8 tokens each (report-only; strict logprob parity not claimed). Attempt 1 ran without `_C` built; kept in `not-green-attempts/`. |
| 1: #543 at `5375881f`, `preflight`, `layers`, `l2` | **passed** (`l2` as a rerun with `--stage l2 --previous`) |
| 2: #544 at `c6bd8466`, `preflight`, `layers`, `l2`, then `bi --bi-suite full` | **passed**; `bi-matrix.json` has **97 cells** |
| 2: chain report and gate for #544 (`--last` is the `l2` run, `--bi` the `bi` run) | report rc 0 (`passed=True first_drift=None cells=16`); `run_qwen3_next_chain_gate.sh` rc 0 (14 passed + report check). Train stages were not run, so the report covers `l2` and `bi` only. CPU-only; no `_C` in that worktree. |
| 3: #557 at `7e312b26`, `--ep 4`, `preflight`, `layers`, `l2` | **passed** |
| 5: #507 attention prior art, `2c0cd1e1`, rerun | **all 10 candidates measured** with the cudnn path set (`step5-rerun/variant2-...`); the first variant without it is kept. Batch invariant yes: rl_kernel_cuda, fa2_split1, triton_2d, flashinfer, megatron_local, te_fused; no: torch_sdpa, fa2_auto, fa4_cute, te_training. max abs vs FP64 6.68e-3 (megatron_local 1.289e-2). TE 2.18.0 / megatron-core 0.16.0rc0 still differ from the B200 reference. |

## Not-green attempts, kept unedited (`not-green-attempts/`, and `run1-...-failed` folders)
1. `chain-544-attempt1-filemode`: launcher `source_mismatch` (file modes).
2. `*-attempt2-launcher-resets-pythonpath`: the launcher resets `PYTHONPATH`, so the preflight smoke aborted again with the 0.16 tvm-ffi.
3. `*-attempt3-...`: #543 preflight failed with `NCCL unhandled cuda error` because another job of the same user was using two GPUs on the node (visible in `header.txt`); #544 got through `preflight` and failed in `layers` on the missing cudnn path.
4. `*-attempt4-mixed-...`: a run on a node shared with a foreign process (18 GiB free), plus overlapping restarts of mine; `ValueError: Free memory ... less than desired`. The dir mixes several of my launches; treat it as an infrastructure failure.
5. `run1-preflight-layers-and-l2-failed` (#543, #544): `l2` failed on the Megatron `CUDA_DEVICE_MAX_CONNECTIONS` assertion.

## Heads
Tested: #543 `5375881f`, #544 `c6bd8466`, #557 `7e312b26`, #551/#507 smoke at `2c699e3f` / `2c0cd1e1`. Current heads on 2026-11-19: #543 `8d45dedc`, #544 `1f41852a`, #551 `eb9c2426`, #557 `7e312b26` (unchanged). The three that moved were **not** retested.

## Drafted "H100" paragraphs (replace those in `SUMMARY.md`)
- **#543** (`5375881f`): 4 x H100 (driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, vLLM 0.30.0), SM90 flags, official checkpoint `9c7f2fbe`, detached clean worktree, no patch: `preflight`, `layers` and `l2` passed. Environment needed `PYTHONNOUSERSITE=1` (apache-tvm-ffi 0.1.11), the cudnn/nccl library path, `CUDA_DEVICE_MAX_CONNECTIONS=1` and `core.fileMode=false`.
- **#544** (`c6bd8466`): same setup: `preflight`, `layers`, `l2` passed; full `bi` suite passed with 97 cells; chain report `passed=True`, gate rc 0 (covers `l2` and `bi`, no train stages).
- **#557** (`7e312b26`): EP4 gate passed (10 cases x 4 ranks); the EP4 chain `preflight`, `layers`, `l2` with `--ep 4` passed.
- **#551** (`2c699e3f`): real 80B rollout smoke with `VLLM_ALLREDUCE_USE_FLASHINFER=0` rc 0; the six probe cases pass with apache-tvm-ffi 0.1.11 and abort with 0.1.16.
- **#507** (`2c0cd1e1`, prior art): 10 of 10 candidates measured after the cudnn path was set; verdicts as above.
