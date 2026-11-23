# H100 results: `main` fixes #746 – #750 and the vLLM sort top-k/top-p fix (handoff 07043486)

Hardware: one node with 4 x NVIDIA H100 (95830 MiB each, driver 580.173.02); every step used GPU 0 only. The node held no other job and no other user process (see `header.txt`, `A..E/other-processes-start.txt`), so D and E timings ran on a GPU with no other job.
Environment: torch 2.13.0+cu130, triton 3.7.1, vLLM 0.30.0, flashinfer-python 0.6.18.post1, Python 3.12.15, nothing installed (`env.txt`). `PYTHONNOUSERSITE=1`. The CUDA toolkit is used for the A build only. B–E ran with an empty `LD_LIBRARY_PATH`, so NVRTC came only from the torch CUDA 13 wheels (`libnvrtc.so.13`, no unversioned name).
Build flags (A): `TORCH_CUDA_ARCH_LIST=9.0 KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 MAX_JOBS=32`, as in the earlier H100 rounds. No tree changes `csrc/` or `setup.py` against `e9d52e95` (`A/csrc-diff.txt` is empty); one `_C` build is shared by all seven trees.
PR heads were re-checked with `gh pr view` before starting: all five equal the README SHAs.
All tracked-file statuses were empty before and after every step (`checks/*.txt`); every `_C` path was inside its own tree.

## Steps

| Step | Result |
|---|---|
| A build | rc=0; all 7 `_C` paths under their own trees; no status lines |
| B full suites | see the table below; no tree has a new failure against H100 `main`; every row of the README's Expected list holds |
| B `main` with the AITER file | 1 collection error in 15 s (rc=2), as expected |
| C revert checks | all four trees: with fix passes, reverted fails, with the same counts as B200 |
| D #747 rms_norm | rc=0; registry choice and all bitwise / batch-invariance flags as expected |
| E top-p vs sort | rc=0; `sort_equals_policy` true for every top_p; Triton differs from the policy on 51–59 of 64 rows |
| F 8-GPU sampling | not run: no 8 x H100 node exists (every H100 node has 4 GPUs) |

## B: per tree (H100)

| Tree | failed | passed | skipped | fixed vs main | new vs main | B200 (failed / passed / skipped) |
|---|---:|---:|---:|---:|---:|---|
| main `e9d52e95` | 72 | 3903 | 906 | - | - | 72 / 3835 / 974 |
| #746 `921f6853` (AITER file runs) | 72 | 3906 | 909 | 0 | 0 | 72 / 3838 / 977 |
| #747 `1f93314b` | 71 | 3909 | 906 | 1 (`test_registry_dispatches_rms_norm`) | 0 | 71 / 3841 / 974 |
| #748 `219ad592` | 70 | 3913 | 906 | 2 (`test_attention_dispatch.py`) | 0 | 70 / 3845 / 974 |
| #749 `acae3a40` | 65 | 3913 | 906 | 7 (`test_reproducible_norm.py` accelerator tests) | 0 | 65 / 3845 / 974 |
| #750 `de996160` | 70 | 3905 | 906 | 2 (`test_canonical_tp_ffn.py::...ipc_reduction[2|64]`) | 0 | not measured |
| sort `95518dbe` | 69 | 3914 | 906 | 3 (`test_captured_full_support_matches_independent_policy[1|4|64]`) | 0 | 69 / 3846 / 974 |

The fixed sets equal B200's for #746 – #749 and sort. Skip counts are lower on H100 (906 vs 974) because the SM90 tests run.

Finding: the baseline failure lists differ between H100 and B200 by 3 tests each, 72 failures on both.
- Only on H100 (3):
  - `tests/integrations/orchestrators/vime/test_vime_tp4_example.py::test_launcher_forces_cuda_graph_without_a_logp_provider`
  - `tests/ops/attention/test_attention_preprocess.py::test_h100_preprocessor_executes_cuda_qk_norm_and_zigzag_rope`
  - `tests/ops/attention/test_attention_preprocess.py::test_h100_preprocessor_is_bitwise_batch_invariant_for_2d_positions`
- Only on B200 (3):
  - `tests/entrypoints/test_cross_config_cli.py::test_run_uses_only_cpu_and_resumes_when_cuda_is_available`
  - `tests/validation/cross_config/test_cross_config_runner.py::test_cpu_smoke_cases_preserve_read_only_scoring_and_exact_provenance`
  - `tests/validation/cross_config/test_cross_config_runner.py::test_runner_resumes_valid_attempt_and_retries_after_identity_or_tensor_change`

These are present in `main` and in every tree alike, so they do not affect any difference between a tree and the baseline. The `-rf` lines carry no message (`--tb=no`), so no reasons are recorded.

## C: targeted tests with the library file reverted to `e9d52e95`

| Tree | Reverted file | with fix | reverted | B200 |
|---|---|---|---|---|
| #747 | `rl_engine/runtime/registry.py` | 68 passed, 72 skipped | 4 failed, 64 passed, 72 skipped | 68 passed / 4 failed |
| #748 | `rl_engine/runtime/registry.py` | 71 passed, 1 skipped | 7 failed, 64 passed, 1 skipped | 71 passed / 7 failed |
| #749 | `rl_engine/ops/norm/reproducible.py` | 17 passed, 1 skipped | 10 failed, 7 passed, 1 skipped | 17 passed / 10 failed |
| sort | `.../vllm/sampling_support.py` | 5 passed | 5 failed | 5 passed / 5 failed |

The reverted copies are plain `cp -a` copies, written with `git show e9d52e95:<file>`; `C/c-file-reverted.txt` shows `_C` and `rl_engine` inside each copy.

## D: #747 native vs CUDA `rms_norm` (bf16, hidden 4096 unless noted)

`registry_choice` is `{"branch": "RMSNormCudaOp", "main": "NativeRMSNormOp"}`. `*_run_to_run_bitwise` and `*_batch_invariant` are true for both ops at all 7 shapes. Ratios are native time / CUDA time (above 1 means CUDA is faster):

| tokens x hidden | forward H100 | forward B200 | fwd+bwd H100 | fwd+bwd B200 | outputs bitwise equal |
|---|---:|---:|---:|---:|---|
| 1 x 4096 | 3.53 | 3.30 | 1.31 | 1.19 | yes |
| 16 x 4096 | 3.38 | 3.37 | 1.60 | 1.22 | yes |
| 256 x 4096 | 3.43 | 3.37 | 1.59 | 1.23 | no (max 7.8e-3) |
| 4096 x 4096 | 7.16 | 8.37 | 0.62 | 0.41 | no (max 1.6e-2) |
| 16384 x 4096 | 7.77 | 6.12 | 0.59 | 0.29 | no (max 1.6e-2) |
| 4096 x 8192 | 7.86 | 6.29 | 0.96 | 0.54 | no (max 1.6e-2) |
| 16384 x 8192 | 8.35 | 6.64 | 0.95 | 0.51 | no (max 1.6e-2) |

The CUDA forward is 3.4 – 8.4x faster on H100 (B200: 3.3 – 8.4x). CUDA forward+backward is faster up to 256 tokens (1.3 – 1.6x), and slower from 4096 tokens (1.0 – 1.7x slower; B200: 2.0 – 3.4x slower). The FP64 error is the same for native and CUDA at every shape.

## E: vLLM 0.30 Triton top-k/top-p vs the sort (64 rows, vocab 151936, temperature 0.7)

| top_p | sort equals policy | rows where Triton differs (H100 / B200) | Triton superset of policy (H100 / B200) | max extra tokens per row (H100 / B200) |
|---|---|---|---|---|
| 0.80 | true | 51 / 60 | true / true | 11 / 10 |
| 0.90 | true | 59 / 54 | true / true | 12 / 13 |
| 0.95 | true | 57 / 55 | **false** / true | 22 / 35 |
| 0.99 | true | 58 / 53 | false / false | 50 / 38 |

Finding: on H100 Triton is not a superset of the sort at p = 0.95 either (B200: only at 0.99).
Batch dependence over 128 cases: Triton 15 (B200: 23), sort 1 (B200: 2).
Cost per call, Triton vs sort, in ms (B200 in brackets):

| rows | Triton | sort |
|---:|---:|---:|
| 1 | 0.180 (0.106) | 0.147 (0.162) |
| 8 | 0.180 (0.120) | 0.434 (0.435) |
| 64 | 0.677 (0.607) | 1.559 (1.044) |
| 256 | 2.165 (1.537) | 5.831 (3.497) |

At one row the sort is cheaper than Triton on H100 (B200: Triton cheaper); from 8 rows Triton is cheaper on both.

## F

Not run: no idle 8 x H100 node exists. The G tree's 8-GPU sampling check is still open.

No failed attempts: nothing was retried, so there is no `not-green-attempts/`.
