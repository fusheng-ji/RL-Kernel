# PR #536 on B200: remaining suites and performance at the exact PR head

Head `54307e1b175943e91ad5f20379003ce73c4dd071`, merge base `16b81175874ecf11d5ff05c00de3b69e9457591b` (built the same way, `TORCH_CUDA_ARCH_LIST=10.0`). Four B200 visible (so the TP=4 case runs), driver 580.126.20, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1. Runner: [run.sh](run.sh). The earlier evidence's source hashes already matched this head (21/21 files); what had run on an older revision were the existing-suite comparison, the multi-GPU cases and the performance experiment, which are repeated here.

| Run | passed | failed | skipped |
|---|---:|---:|---:|
| `test_det_gemm.py`, tuning off / on | 73 / 73 | 0 / 0 | 9 / 9 |
| HAL + registry | 34 | 0 | 0 |
| operator, cache and graph regressions | 60 | 0 | 2 |
| candidate qualification | 402 | 0 | 27 |
| benchmark functionality | 18 | 0 | 0 |
| `tests/ops`, head, tuning off | 2044 | 2 | 401 |
| `tests/ops`, head, tuning on | 2045 | 1 | 401 |
| `tests/ops`, merge base, tuning off | 1532 | 1 | 412 |
| `tests/ops`, merge base, tuning on | 1530 | 3 | 412 |

Failures (names from the logs): `test_registry_dispatches_rms_norm` fails in all four runs (also on the merge base); `test_autograd_wrapper_contiguity_and_gradient_routing[...]` (CPU, unseeded) fails once at the head with tuning off and twice on the merge base with tuning on, i.e. flaky on both trees. Nothing new at the head. More tests run at the head than at the merge base because the GEMM import guard fix makes the Triton GEMM cases run.

Performance (`results/issue50-performance.json`, 52 cases, one process group of 5 trials; the original used 5 process groups, so the "tuned slower" counts are not comparable): confirmed benefit eager 0, CUDA graph 19, as in the PR body.

Draft addition for the PR body Results:

> **Re-run at head `54307e1b` (B200, torch 2.13, 4 GPUs visible).** All six PR commands pass with the same counts as the table above (73/9, 34, 60/2, 402/27, 18). `tests/ops`: head 2044 passed / 2 failed / 401 skipped (tuning off), 2045 / 1 / 401 (on); merge base 1532 / 1 / 412 and 1530 / 3 / 412. The failures are `test_registry_dispatches_rms_norm` (all runs, both trees) and the flaky CPU `swiglu` autograd test. Performance, 52 cases: 0 eager and 19 CUDA-graph cases with a confirmed benefit. Raw data: `pr_assets` `pr536/b200-gaps/` @ `<EVIDENCE_SHA>`.
