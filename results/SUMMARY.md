# Qwen3-Next: stock VIME replay at its recipe layout (H100, 2026-11-22)

Handoff: `handoff/h100-qwen3-next-stock-tp2-20261121` (`daa42a48`). Code: `37e04258` (bundle ref `local/e2e-bench-gdn-fused-tp2`), worktree clean before and after every step (`*/status-after.txt` empty).
Hardware: one node, 4 x NVIDIA H100 (95830 MiB), driver 580.173.02 (`nvidia-smi-name-driver.csv`), idle at the start of each step (`*/nvidia-smi-start.txt`).
Environment: last round's, unchanged; nothing installed (`pip-list-filtered.txt`: torch 2.13.0+cu130, vllm 0.30.0, flashinfer-python 0.6.18.post1, transformer_engine 2.18.0, megatron-core 0.16.0rc0, flash-linear-attention / fla-core 0.4.2, apache-tvm-ffi 0.1.11). `PYTHONNOUSERSITE=1`, `CUDA_DEVICE_MAX_CONNECTIONS` unset, `CUDNN_HOME` and `LD_LIBRARY_PATH` at the base environment's nvidia directories (`*/header.txt`).

| Step | Result |
|---|---|
| A build | **passed**: `build.log` rc 0; `c-file.txt` inside the worktree; tree clean |
| B TP4 chain + `bi` full | **passed**: `preflight`, `layers`, `l2`, `bi` all `passed`, each `source_commit` `37e04258`; `bi-matrix.json` 97 cells, 97 passed; chain report `passed=True first_drift=None family=None cells=16`; gate (`PY=<PY>`) `17 passed`, rc 0. Same as last round's B. |
| C bench, rollout + replay | **passed**: `rc=0`; `rl_kernel_commit` `37e04258…`, `tracked_tree_dirty: false`; no replay or rollout `failures`; replay layouts as expected; only `native_bi` unsupported; every strict rollout check true or null; strict times within about 2.5% of last round's |
| D CPU suite | **as expected**: 15 failed, 663 passed (last round 15 / 658, plus the 5 new tests); all 15 are `test_qwen3_next_training_checkpoint.py` with `AssertionError: Invalid device id` |

Wall times: A 1.5 min, B 2 h 02 min (00:31Z to 02:33Z), C 3 h 25 min (02:33Z to 05:58Z), D 1.5 min (`steps.txt`).

## C details

- `phases.replay.layouts`: `strict` `{"tp": 4, "sp": false, "ep": 1, "etp": 4, "dp": 1}`, `native` `{"tp": 2, "sp": true, "ep": 4, "etp": 1, "dp": 2}`.
- `phases.replay`: `strict` and `native` results for both workloads, with `strict_over_native`; no `failures` key.
- `phases.rollout.unsupported`: only `native_bi`, for both workloads, "VLLM batch_invariant mode is not supported for GDN_ATTN.".
- Strict rollout: `same_tokens_across_repeats`, `same_tokens_across_rounds` and `row0_equals_batch1` true on every shape, or null where they do not apply (batch 1).

Replay medians (`forward_logprob`, seconds):

| workload | shape | strict (TP4), this run | strict, last round H100 | stock VIME (TP2 + SP, EP4, DP2) | strict / stock |
|---|---|---:|---:|---:|---:|
| certified | p64-b1 | 0.282 | 0.285 | 0.426 | 0.66 |
| certified | p64-b4 | 0.310 | 0.309 | 0.471 | 0.66 |
| certified | p256-b1 | 0.290 | 0.287 | 0.463 | 0.63 |
| certified | p256-b4 | 0.319 | 0.322 | 0.491 | 0.65 |
| certified | p1024-b1 | 0.420 | 0.420 | 0.476 | 0.88 |
| certified | p1024-b4 | 0.573 | 0.575 | 0.502 | 1.14 |
| long | p512-b1-r512 | 0.423 | 0.425 | 0.456 | 0.93 |
| long | p512-b4-r512 | 0.573 | 0.578 | 0.491 | 1.17 |
| long | p512-b16-r512 | 1.448 | 1.450 | 0.535 | 2.71 |

Rollout totals (`total_s`, seconds):

| workload | shape | strict, this run | strict, last round H100 | native graph | native eager |
|---|---|---:|---:|---:|---:|
| certified | p64-b1 | 4.52 | 4.56 | 0.10 | 1.75 |
| certified | p64-b4 | 5.16 | 5.15 | 0.24 | 1.99 |
| certified | p256-b1 | 4.53 | 4.56 | 0.18 | 1.74 |
| certified | p256-b4 | 5.21 | 5.27 | 0.32 | 1.99 |
| certified | p1024-b1 | 4.68 | 4.72 | 0.17 | 1.73 |
| certified | p1024-b4 | 5.57 | 5.62 | 0.31 | 1.97 |
| long | p512-b1-r512 | 163.20 | 160.79 | 2.26 | 54.85 |
| long | p512-b16-r512 | 217.24 | 213.57 | 3.50 | 56.25 |
| long | p512-b64-r512 | 370.46 | 361.93 | 5.93 | 57.06 |

## Notes

- After A, my wrapper (`scripts/run3.sh.txt`) stopped with "c-file not inside worktree: stop" (`steps.txt`): it compared `c-file.txt` with the worktree path without resolving symlinks. A itself had passed (rc 0, `_C` inside the worktree). B, C and D ran from `scripts/run3_from_b.sh.txt`, which resolves both paths; no step was repeated, and this was not a test failure.
- Hosts are written `node-a`; user, paths, job ids and environment names are redacted.
