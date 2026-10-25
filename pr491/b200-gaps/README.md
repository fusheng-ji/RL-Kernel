# PR #491 on B200: eight-GPU tests and benchmark at the PR head

Head `cc27cf5cb565857abc1d78ae77d1541872e820d1`; "before" = merge base `5ba769f681b7fa765ea6b29f9939454a858390ba` (same build flags, `TORCH_CUDA_ARCH_LIST=10.0`) run with the head's benchmark script. Eight B200 of one node, driver 580.126.20, CUDA 13.0, torch 2.13.0+cu130. Runners: [run.sh](run.sh), [run-w8.sh](run-w8.sh).

| Run | passed | failed | skipped |
|---|---:|---:|---:|
| all-gather size/alignment sweep + all-gather + transport tests, 8 GPUs | 41 | 0 | 0 |
| same, 2 GPUs visible | 40 | 0 | 1 (needs 8 GPUs) |
| `test_deterministic_all_reduce.py`, `test_deterministic_reduce_scatter.py`, 8 GPUs | 1 | 1 | 0 |

The failing test is `test_deterministic_reduce_scatter_cross_tp_cuda`: `reduce_scatter_many() got an unexpected keyword argument 'outs'`, as the PR body already states for unmodified main. I did not re-run these two files on the merge base in this round.

Benchmark, 2 ranks (`results/{before,after}_w2.json`), all_gather median us: 8 KiB 81.0 -> 23.7, 32 KiB 266.9 -> 37.6, 128 KiB 1008.7 -> 36.4, 1 MiB 52.1 -> 41.6, 4 MiB 86.9 -> 50.4 (NCCL 22.7 / 23.2 / 25.2 / 31.3 / 45.4).

Benchmark, 8 ranks. The full 8-rank run (`results/bench-{before,after}-w8.log`) aborts in `all_reduce` at the same point on the merge base and on the head ("CUDA error: too many resources requested for launch", from `deterministic_collective_all_reduce_fused`, a kernel this PR does not touch; `results-w8-rerun/` repeats it for all_reduce alone). So the 8-rank table is restricted to the operations the PR changes (`results-w8-rerun/{before,after}_w8_all_gather.json`):

| per-rank input | before | after | NCCL |
|---|---:|---:|---:|
| 1 KiB | 94.0 us | 48.5 | 34.2 |
| 8 KiB | 456.5 | 71.5 | 35.5 |
| 32 KiB | 1689.8 | 58.7 | 38.8 |
| 128 KiB | 70.6 | 153.4 (outlier, NCCL also 220.9 in that sample) | 220.9 |
| 1 MiB | 137.1 | 86.1 | 62.2 |
| 4 MiB | 347.7 | 142.3 | 150.6 |

`all_gather_many` is unchanged (e.g. 4 MiB 220.0 -> 220.0 us). The 64 KiB crossover constant is consistent with these rows; no retuning was attempted.

Draft addition for the PR body Results:

> **Eight B200 at head `cc27cf5c`.** The 8-GPU all-gather tests pass (41 passed, 0 skipped); on a 2-GPU view 40 passed / 1 skipped. `test_deterministic_reduce_scatter_cross_tp_cuda` still fails with the `outs=` TypeError described above. 8-rank all-gather, merge base -> head: 8 KiB/rank 456.5 -> 71.5 us, 32 KiB 1689.8 -> 58.7 us, 1 MiB 137.1 -> 86.1 us, 4 MiB 347.7 -> 142.3 us (NCCL 150.6). The full 8-rank benchmark aborts in the fused `all_reduce` ("too many resources requested for launch") on both trees, so only all_gather / all_gather_many rows are reported. Raw data: `pr_assets` `pr491/b200-gaps/` @ `<EVIDENCE_SHA>`.
