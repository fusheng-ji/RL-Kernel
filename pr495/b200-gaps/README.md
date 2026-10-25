# PR #495 on B200: tests and evidence rerun at the PR head

Head `e612412c815d02d4d05dac1c5fe44d4688dcc82c` (clean tree, extension built `TORCH_CUDA_ARCH_LIST=10.0`, fast-math flag unset), B200, driver 580.126.20, CUDA 13.0, torch 2.13.0+cu130. Runner: [run.sh](run.sh). liger-kernel 0.8.4 and fla-core 0.5.2 came from wheels unpacked into a separate directory (not installed into the shared environment); flash-attention source at `94e22c9`.

| Run | passed | failed | skipped |
|---|---:|---:|---:|
| 14 test files (the PR set at current paths, plus indexing/INT_MAX and the two benchmark-evidence files), 2 GPUs visible | 371 | 0 | 34 |
| same, 1 GPU visible | 369 | 0 | 36 (the 2 extra are the two-device regression: "two CUDA devices are required") |

Skips are SM90/Hopper-only kernels and comparisons, Ascend, and one CUDA-dispatch guard. The device-guard multi-GPU regression and the INT_MAX loop boundary test (`test_fused_logp_backward_indexing.py`) ran and passed.

`benchmarks/fused_logp_backward_evidence.py` at the head (`results/report.json`, `git_commit` = head, `git_dirty=false`): shared-gradient accuracy, row invariance and batch-size sweep as in the review rerun (same max abs gradient error 3.2134e-3 for all candidates; fused kernel 99.9986% correctly rounded; FLA CE differs in 92/2232 batch comparisons, all others 0).

**Timing at the head differs from the PR body.** 32768 rows x 151936, BF16, forward + backward / backward only:

| | PR body (measured before the int64 loop fix) | head |
|---|---|---|
| fused kernel | 27.2 / 12.5 ms | **33.2 / 28.4 ms** |
| chunked fallback | 59.4 / 44.8 ms | 63.7 / 58.9 ms |
| torch log_softmax + gather | 44.0 / 19.3 ms | 44.4 / 19.4 ms |
| Liger CE | 15.8 / 11.3 ms | 16.2 / 11.6 ms |
| FLA CE | 7.1 / 3.7 ms | 7.1 / 3.8 ms |
| flash-attn CE | 6.2 / 3.7 ms | 6.2 / 3.7 ms |

Peak memory is unchanged (9.27 GiB). The unchanged rows show the run was not disturbed; only the fused backward got slower, 2.3x (backward only), consistent with the 64-bit loop counters added for the INT_MAX fix. The PR body table, the "27.2 ms" prior-art row and the "4.4x / 3.4x slower than flash-attn" statements should be refreshed (now 5.4x / 7.7x).

Draft addition for the PR body Results:

> **Re-measured at head `e612412c` (B200, torch 2.13).** After the device guard and the 64-bit loop counters, the fused backward takes 28.4 ms (backward only) and 33.2 ms (forward + backward) at 32768 rows, against 12.5 / 27.2 ms before; memory is unchanged at 9.27 GiB. The chunked fallback (63.7 ms), torch (44.4 ms), Liger (16.2 ms), FLA (7.1 ms) and flash-attn (6.2 ms) rows reproduce the earlier numbers. Tests: 371 passed / 34 skipped on two GPUs (369 / 36 on one), including the multi-GPU device-guard and INT_MAX regressions. Shared-gradient accuracy is unchanged. Raw data: `pr_assets` `pr495/b200-gaps/` @ `<EVIDENCE_SHA>`.
