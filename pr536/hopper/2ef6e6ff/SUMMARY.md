# PR #536 on H100 — evidence at head `2ef6e6ff`, compared against `upstream/main` @ `49941801`

## 0. Header

| item | value |
|---|---|
| GPUs | 4 x NVIDIA H100 (sm90, 95830 MiB each), all 4 visible (`CUDA_VISIBLE_DEVICES=0,1,2,3`) |
| driver / CUDA | 580.173.02 / 13.0 |
| python / torch / triton | 3.12.15 / 2.13.0+cu130 / 3.7.1 (`<HOME>/conda_envs/rlk_ep4/...`, same env for all runs, nothing installed) |
| head worktree | detached at `2ef6e6ffbd338ede917e629176754150584cdaaa` (PR head confirmed with `gh pr view 536 --json headRefOid` right before starting) |
| main worktree | detached at `49941801f942428fd52ac5b66a1b0c070418724e` (`RL-Align/RL-Kernel` main at the time) |
| sm90-gemm worktree | detached at `2ef6e6ffbd338ede917e629176754150584cdaaa`, separate build, see section 4 |
| build | `TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace` for head and main; sm90-gemm additionally `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1` |
| `_C` location | `c-file.txt` in each result directory; each points inside its own worktree |
| env | `PYTHONNOUSERSITE=1 PYTHONPATH=. RL_KERNEL_MODE=auto`, a fresh `RL_KERNEL_AUTOTUNE_CACHE_DIR=$(mktemp -d)` per command (default `/tmp`), `PATH` with `rlk_ep4/bin` and `cuda13_home/bin` |
| git status | `git status --porcelain --untracked-files=no` is **empty before and after every step** (`status-before.txt` / `status-after.txt`). The shared filesystem drops the executable bit, which shows up as 21 mode-only changes (`100755 => 100644`, 0 lines changed) unless `core.fileMode=false` is set; I ran with `core.fileMode=false` via environment (`node/mode-only-note.txt`). No file was edited. |
| other job on the node | No. The allocation was used serially, only on one node (node A) for all GPU steps. At the start `nvidia-smi` showed 0 MiB used and 0 compute processes on all 4 GPUs (`node/nvidia-smi-before.txt`, `header.txt`), and `squeue` for the node listed only this allocation (`node/squeue-node.txt`). The two other builds (head/main in sequence on node A; sm90-gemm on the second node of the allocation) were CPU-only compilation, no GPU was used by them, and finished before any GPU step started. |

Scripts used are in `results/scripts/` (paths redacted).

### Deviation that I made and corrected (kept, not hidden)
My first run of step 2 used a `TMPDIR` on the shared filesystem (setgid directory, mode 2700). The autotuner then logged `Autotune cache ... is not private (writable by group or others); using memory` and 19 tests failed (the restart-misses-disk-cache family). That was a harness mistake, not a result about the PR, since the plan asks for `mktemp -d` under the default temp dir. I removed `TMPDIR`, and re-ran step 2 and every later step with the default `/tmp`. The first attempt is kept in `pr536/2ef6e6ff/autotune-shared-tmpdir-DEVIATION/` with a note. Nothing else was re-run: steps 3 and 4 were only ever run once, with the correct environment.

## 1. Build

- head, main, sm90-gemm all built with rc=0 (`build.log`).
- `c-file.txt` for each shows the `_C` shared object inside its own worktree.
- triton version/path per worktree is in `header.txt` (3.7.1, same env).

## 2. Autotune suite on head (`2ef6e6ff`)

`pytest -p no:cacheprovider -q -rs tests/platforms/test_persistent_autotuner.py tests/runtime/test_kernel_registry.py tests/ops/test_autotuning_gpu.py tests/ops/test_autotuning_candidates.py tests/benchmarks/test_autotuning_benchmark.py`

**Result: 524 passed, 29 skipped, 0 failed** (37 s). Skips: 2 "operator does not support this dtype" (`test_autotuning_gpu.py:95`), 27 "unsupported dtype" (`test_autotuning_candidates.py`). Log: `pr536/2ef6e6ff/autotune/suite.log`.

`pytest -v -rs tests/platforms/test_persistent_autotuner.py -k out_of_memory` -> **2 passed, 30 deselected**. Both new tests appear by name (`autotune/oom-v.log`):

- `test_launch_falls_back_when_profiling_runs_out_of_memory` PASSED
- `test_launch_propagates_profiling_errors_that_are_not_out_of_memory` PASSED

## 3. `tests/ops`, one file per process (head vs `49941801`, AUTOTUNE=0 and 1)

Each cell is passed/failed/skipped/error (process rc). No file hung; there were no timeouts. The per-file logs and `rc.txt` are in `ops-autotune0/` and `ops-autotune1/` under each result directory. 36 files on head, 34 on main.

### RL_KERNEL_AUTOTUNE=0

| file | head P/F/S/E (rc) | main P/F/S/E (rc) | verdict |
|---|---|---|---|
| `tests/ops/activation/test_final_logit_softcap.py` | 160/0/0/0 (0) | 160/0/0/0 (0) | same |
| `tests/ops/activation/test_swiglu.py` | 129/6/48/0 (1) | 135/0/48/0 (0) | **head only** (fail) |
| `tests/ops/attention/test_attention.py` | 27/0/0/0 (0) | 27/0/0/0 (0) | same |
| `tests/ops/attention/test_attention_correctness.py` | 50/0/82/0 (0) | 50/0/82/0 (0) | same |
| `tests/ops/attention/test_attention_cross_config_binding.py` | 81/0/0/0 (0) | 81/0/0/0 (0) | same |
| `tests/ops/attention/test_attention_preprocess.py` | 5/0/2/0 (0) | 5/0/2/0 (0) | same |
| `tests/ops/attention/test_attention_projection.py` | 8/0/0/0 (0) | 8/0/0/0 (0) | same |
| `tests/ops/attention/test_deterministic_attention_cuda.py` | 612/0/29/0 (0) | 612/0/29/0 (0) | same |
| `tests/ops/attention/test_kv_cache_attention.py` | 15/0/0/0 (0) | 15/0/0/0 (0) | same |
| `tests/ops/attention/test_triton_batch_invariant_attention.py` | 35/0/0/0 (0) | 35/0/0/0 (0) | same |
| `tests/ops/attention/test_triton_deterministic_attention.py` | 1/0/70/0 (0) | 1/0/70/0 (0) | same |
| `tests/ops/autograd/test_vjp_fp32.py` | 50/0/0/0 (0) | 50/0/0/0 (0) | same |
| `tests/ops/embedding/test_canonical_embedding.py` | 5/0/0/0 (0) | 5/0/0/0 (0) | same |
| `tests/ops/embedding/test_canonical_lm_head_backward.py` | 12/0/0/0 (0) | 12/0/0/0 (0) | same |
| `tests/ops/embedding/test_embedding.py` | 11/0/0/0 (0) | 11/0/0/0 (0) | same |
| `tests/ops/embedding/test_issue151_embedding_lm_head_invariance.py` | 9/0/0/0 (0) | 9/0/0/0 (0) | same |
| `tests/ops/embedding/test_lm_head.py` | 18/0/0/0 (0) | 18/0/0/0 (0) | same |
| `tests/ops/embedding/test_triton_embedding.py` | 32/0/0/0 (0) | 32/0/0/0 (0) | same |
| `tests/ops/gemm/test_det_gemm.py` | 55/18/9/0 (1) | 20/2/49/0 (1) | **head only** (fail) |
| `tests/ops/gemm/test_matmul.py` | 21/0/0/0 (0) | 21/0/0/0 (0) | same |
| `tests/ops/linear_attn/test_gdn_state_contract.py` | 35/0/0/0 (0) | 35/0/0/0 (0) | same |
| `tests/ops/logprob/test_batch_invariant_logp.py` | 56/0/30/0 (0) | 56/0/30/0 (0) | same |
| `tests/ops/logprob/test_deterministic_logp.py` | 74/0/0/0 (0) | 74/0/0/0 (0) | same |
| `tests/ops/logprob/test_linear_logp.py` | 33/0/20/0 (0) | 33/0/20/0 (0) | same |
| `tests/ops/logprob/test_logp.py` | 17/0/1/0 (0) | 17/0/1/0 (0) | same |
| `tests/ops/loss/test_grpo_loss.py` | 16/0/0/0 (0) | 16/0/0/0 (0) | same |
| `tests/ops/loss/test_ratio_kl.py` | 150/0/0/0 (0) | 150/0/0/0 (0) | same |
| `tests/ops/norm/test_reproducible_norm.py` | 7/8/0/0 (1) | 7/8/0/0 (1) | same |
| `tests/ops/norm/test_rms_norm.py` | 50/1/72/0 (1) | 50/1/72/0 (1) | same |
| `tests/ops/packing/test_pack.py` | 19/0/0/0 (0) | 19/0/0/0 (0) | same |
| `tests/ops/rope/test_rope.py` | 26/0/7/0 (0) | 26/0/7/0 (0) | same |
| `tests/ops/sampling/test_complete_sampling.py` | 9/0/0/0 (0) | 9/0/0/0 (0) | same |
| `tests/ops/sampling/test_sampler_temperature.py` | 2/0/0/0 (0) | 2/0/0/0 (0) | same |
| `tests/ops/test_op_accuracy.py` | 32/0/2/0 (0) | 32/0/2/0 (0) | same |

Files only on head: ['tests/ops/test_autotuning_candidates.py', 'tests/ops/test_autotuning_gpu.py']

Files only on main: []


### RL_KERNEL_AUTOTUNE=1

| file | head P/F/S/E (rc) | main P/F/S/E (rc) | verdict |
|---|---|---|---|
| `tests/ops/activation/test_final_logit_softcap.py` | 160/0/0/0 (0) | 160/0/0/0 (0) | same |
| `tests/ops/activation/test_swiglu.py` | 129/6/48/0 (1) | 135/0/48/0 (0) | **head only** (fail) |
| `tests/ops/attention/test_attention.py` | 27/0/0/0 (0) | 27/0/0/0 (0) | same |
| `tests/ops/attention/test_attention_correctness.py` | 50/0/82/0 (0) | 50/0/82/0 (0) | same |
| `tests/ops/attention/test_attention_cross_config_binding.py` | 81/0/0/0 (0) | 81/0/0/0 (0) | same |
| `tests/ops/attention/test_attention_preprocess.py` | 5/0/2/0 (0) | 5/0/2/0 (0) | same |
| `tests/ops/attention/test_attention_projection.py` | 8/0/0/0 (0) | 8/0/0/0 (0) | same |
| `tests/ops/attention/test_deterministic_attention_cuda.py` | 612/0/29/0 (0) | 612/0/29/0 (0) | same |
| `tests/ops/attention/test_kv_cache_attention.py` | 15/0/0/0 (0) | 15/0/0/0 (0) | same |
| `tests/ops/attention/test_triton_batch_invariant_attention.py` | 35/0/0/0 (0) | 35/0/0/0 (0) | same |
| `tests/ops/attention/test_triton_deterministic_attention.py` | 1/0/70/0 (0) | 1/0/70/0 (0) | same |
| `tests/ops/autograd/test_vjp_fp32.py` | 50/0/0/0 (0) | 50/0/0/0 (0) | same |
| `tests/ops/embedding/test_canonical_embedding.py` | 5/0/0/0 (0) | 5/0/0/0 (0) | same |
| `tests/ops/embedding/test_canonical_lm_head_backward.py` | 12/0/0/0 (0) | 12/0/0/0 (0) | same |
| `tests/ops/embedding/test_embedding.py` | 11/0/0/0 (0) | 11/0/0/0 (0) | same |
| `tests/ops/embedding/test_issue151_embedding_lm_head_invariance.py` | 9/0/0/0 (0) | 9/0/0/0 (0) | same |
| `tests/ops/embedding/test_lm_head.py` | 18/0/0/0 (0) | 18/0/0/0 (0) | same |
| `tests/ops/embedding/test_triton_embedding.py` | 32/0/0/0 (0) | 32/0/0/0 (0) | same |
| `tests/ops/gemm/test_det_gemm.py` | 55/18/9/0 (1) | 20/2/49/0 (1) | **head only** (fail) |
| `tests/ops/gemm/test_matmul.py` | 21/0/0/0 (0) | 21/0/0/0 (0) | same |
| `tests/ops/linear_attn/test_gdn_state_contract.py` | 35/0/0/0 (0) | 35/0/0/0 (0) | same |
| `tests/ops/logprob/test_batch_invariant_logp.py` | 56/0/30/0 (0) | 56/0/30/0 (0) | same |
| `tests/ops/logprob/test_deterministic_logp.py` | 74/0/0/0 (0) | 74/0/0/0 (0) | same |
| `tests/ops/logprob/test_linear_logp.py` | 33/0/20/0 (0) | 33/0/20/0 (0) | same |
| `tests/ops/logprob/test_logp.py` | 17/0/1/0 (0) | 17/0/1/0 (0) | same |
| `tests/ops/loss/test_grpo_loss.py` | 16/0/0/0 (0) | 16/0/0/0 (0) | same |
| `tests/ops/loss/test_ratio_kl.py` | 150/0/0/0 (0) | 150/0/0/0 (0) | same |
| `tests/ops/norm/test_reproducible_norm.py` | 7/8/0/0 (1) | 7/8/0/0 (1) | same |
| `tests/ops/norm/test_rms_norm.py` | 50/1/72/0 (1) | 50/1/72/0 (1) | same |
| `tests/ops/packing/test_pack.py` | 19/0/0/0 (0) | 19/0/0/0 (0) | same |
| `tests/ops/rope/test_rope.py` | 26/0/7/0 (0) | 26/0/7/0 (0) | same |
| `tests/ops/sampling/test_complete_sampling.py` | 9/0/0/0 (0) | 9/0/0/0 (0) | same |
| `tests/ops/sampling/test_sampler_temperature.py` | 2/0/0/0 (0) | 2/0/0/0 (0) | same |
| `tests/ops/test_op_accuracy.py` | 32/0/2/0 (0) | 32/0/2/0 (0) | same |

Files only on head: ['tests/ops/test_autotuning_candidates.py', 'tests/ops/test_autotuning_gpu.py']

Files only on main: []


### Head-only differences
1. `tests/ops/activation/test_swiglu.py`: **6 failed on head, 0 on main**, identical at AUTOTUNE=0 and 1. All 6 are `test_triton_swiglu_launchers_use_input_device_and_stream[{0,1}-{dtype0,dtype1,dtype2}]`. First traceback:
```
kernel = <rl_engine.backends.shared.triton.autotuning.TunableKernel object at 0x...>

    def record_launch(kernel):
>       original_run = kernel.run
                       ^^^^^^^^^^
E       AttributeError: 'TunableKernel' object has no attribute 'run'

tests/ops/activation/test_swiglu.py:675: AttributeError
```
   (called from `tests/ops/activation/test_swiglu.py:688: record_launch(kernel)`). The test monkeypatches `kernel.run` on the Triton kernels; on head `swiglu._swiglu_fwd_kernel`, `_swiglu_bwd_kernel`, `_swiglu_dgate_kernel` are `TunableKernel` wrappers that have no `run`. The test is guarded by `torch.cuda.device_count() >= 2`, so it is only exercised on a node with at least 2 GPUs.
2. `tests/ops/gemm/test_det_gemm.py`: head 55/18/9 vs main 20/2/49. This is **not a new crash**, see the next two points.
   - **The 40 "Triton is unavailable" skips of main do not reappear on head.** On main they come from the `try: ... except ImportError` guard in the test file (`_HAS_TRITON = False`, because the file imports `_det_gemm_tree_reduce_to_output_kernel`). On head the PR changed the test (it imports `_det_gemm_tree_reduce_to_output_rocm_kernel`, and only a missing `triton` module disables the tests), so the Triton parametrizations now run on H100. The only skips left are the 9 ROCm gfx942 ones (same 9 as main).
   - The first failure is the same on both sides: `test_target_shapes_invariance[shape3-cuda-deterministic_gemm]` (shape (4096, 14336, 4096)), `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`. On main the second failure is shape4-cuda (also the sticky error). On head all 18 failures have this same error message (18 of 18 `E` lines), and they are the tests that run after the sticky error in the same process, plus the Triton parametrizations that main skips. See section 4 for the isolated run.

### Same on both sides (not caused by the PR)
- `tests/ops/norm/test_reproducible_norm.py`: 8 failed on both, `OSError: libnvrtc.so.12: cannot open shared object file` (the env has CUDA 13 runtime libs).
- `tests/ops/norm/test_rms_norm.py`: 1 failed on both, `test_registry_dispatches_rms_norm`: `assert isinstance(op, RMSNormCudaOp)` is False.

### Files only on one side
- Only on head: `tests/ops/test_autotuning_candidates.py`, `tests/ops/test_autotuning_gpu.py` (they are added by the PR; they are the same files used in step 2, where they pass).
- Only on main: none.

## 4. Crash test `test_target_shapes_invariance`, alone

`CUDA_LAUNCH_BLOCKING=1 pytest -p no:cacheprovider -v -rs "tests/ops/gemm/test_det_gemm.py::test_target_shapes_invariance"`

| build | result | log |
|---|---|---|
| head `2ef6e6ff` (default build) | **4 failed, 6 passed**. shape0-2 pass for cuda and triton; shape3-cuda FAILS first (illegal memory access, test line 491); the later shape3-triton, shape4-cuda, shape4-triton fail because the CUDA context is poisoned (the traceback of shape3-triton is inside `torch.manual_seed`) | `pr536/2ef6e6ff/crash/crash-test.log` |
| main `49941801` (default build) | **2 failed, 3 passed**. shape0-2 pass; shape3-cuda FAILS first (illegal memory access, test line 488), then shape4-cuda | `main-49941801/crash/crash-test.log` |
| head `2ef6e6ff`, **separate build with `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1`** | **10 passed** | `pr536/2ef6e6ff-sm90-gemm/crash/crash-test.log` |

In the sm90-gemm build the whole `tests/ops/gemm/test_det_gemm.py` file: **73 passed, 9 skipped, 0 failed** with `CUDA_LAUNCH_BLOCKING=1` (`whole-det-gemm.log`), and again **73 passed / 9 skipped** without `CUDA_LAUNCH_BLOCKING` (`whole-det-gemm-noblocking.log`). The 9 skips are the ROCm gfx942 ones.

Conclusion: the illegal memory access at shape (4096, 14336, 4096) in the native CUDA `deterministic_gemm` is the **same on head and main** with the default build, so it is not introduced by PR #536. It disappears completely when the extension is built with the SM90 det-GEMM flags (reported as a separate build, not as the PR's default result).

## 5. Draft "H100" paragraph for the PR description

> **H100 (4 x NVIDIA H100, driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1).** At `2ef6e6ff`, the autotune suite (`test_persistent_autotuner`, `test_kernel_registry`, `test_autotuning_gpu`, `test_autotuning_candidates`, `test_autotuning_benchmark`) gives 524 passed, 29 skipped, 0 failed, and both new tests `test_launch_falls_back_when_profiling_runs_out_of_memory` and `test_launch_propagates_profiling_errors_that_are_not_out_of_memory` pass. `tests/ops` was run one file per process, with `RL_KERNEL_AUTOTUNE=0` and `=1`, on `2ef6e6ff` and on `upstream/main` @ `49941801`, and the results are identical for both autotune settings. Compared with `upstream/main` @ `49941801`, 32 of 34 shared files have the same counts at head, except `tests/ops/activation/test_swiglu.py` (6 failures only at `2ef6e6ff`: `test_triton_swiglu_launchers_use_input_device_and_stream` does `kernel.run` on a `TunableKernel`, which has no `run`; needs >= 2 GPUs) and `tests/ops/gemm/test_det_gemm.py` (the 40 Triton tests skipped on `upstream/main` run at `2ef6e6ff`; the illegal-memory-access crash at `test_target_shapes_invariance[shape3-cuda-deterministic_gemm]` happens on both sides and is fixed by building with `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1`, with all of `test_det_gemm.py` passing, 73 passed, 9 skipped, at `2ef6e6ff`). `test_reproducible_norm.py` (8, missing `libnvrtc.so.12`) and `test_rms_norm.py` (1) fail identically at `upstream/main` @ `49941801`.

## 6. Unexpected / worth knowing
- `test_swiglu.py` fails 6 times at head only (section 3); no test in the autotune suite covers it.
- Because of the sticky CUDA error, head's `test_det_gemm.py` shows 18 failures vs 2 on main in one process; the extra 16 are the Triton parametrizations that main skipped, plus tests after the poisoned context. This is not evidence of a new kernel bug, but it can't be separated further without running those tests in separate processes; I did not do that (outside the plan).
- Step 2's first attempt with a shared-FS `TMPDIR` (section 0).
