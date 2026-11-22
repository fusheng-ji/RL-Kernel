# H3 final run, PR A (#731 on #730): the Hopper row (H100, 2026-11-22)

Handoff: `handoff/h100-h3-final-run-pra-20261120` at `fe2ea386` (README updated 2026-11-22).
Code: head #731 `b446400f` (`<SRC>`), base #730 `02b123e0` (`<BASE>`), both fetched from the PRs; `commit.txt` in every step.
Hardware: NVIDIA H100 80GB HBM3, 4 per node. Steps A, A2, B, D and the 2- and 4-rank C on `node-a`; the 8-rank C on `node-a` + `node-b`.
Each step folder has `commit.txt`, `nvidia-smi*.txt`, `env.txt` (the environment line for `<E1>` and `<E2>`), the tree status before and after (all empty) and the user's other processes before and after.

Environments:
- `<E1>`: torch 2.13.0+cu130, triton 3.7.1 (`pip-list-filtered-E1.txt`), with the H3 provider (Diffusers with MiniMax-H3) on `PYTHONPATH` where the suites need it.
- `<E2>`: torch 2.14.0+cu130, triton 3.8.0, diffusers 0.43.0, transformers 5.19.0, peft 0.22.0, accelerate 1.16.0, verl 0.10.0.dev0, verl-omni 0.4.0 at `a634f4d8` (`pip-list-filtered-E2.txt`).

Verdict: **every expectation is met except the ones the README itself predicts for this machine** (the base side of the Megatron hook and 8 base accelerator tests need `libnvrtc.so.12`, and this environment has only the CUDA 13 NVRTC). Nothing fails only at head.

| Step | Result |
|---|---|
| A timing | done; table below |
| A norm partitions | `exact_bitwise_equal` 16 / 16, `torch_bitwise_equal` 15 / 16 (same as B200) |
| A Megatron hook | rc 1: the base side fails with `OSError: libnvrtc.so.12: cannot open shared object file` (the bug PR A fixes); no `megatron-hook.json` |
| A2 row-mean bytes | `all_bytes_equal: true` (16 cases, compared with `row_mean_bytes_b200.json`); `test-row-mean.txt` 22 passed |
| B verl-omni compare | 560 / 560 on `prev_sample`, `prev_sample_mean`, `std_dev_t`, `sqrt_dt`; `log_prob` 206 / 560, max abs 1.788e-07 |
| B log-prob batch | `rl_kernel_differ` 0, `verl_omni_differ` 33 of 144 (B200: 27) |
| B transition tests | 9 passed |
| C TP2, TP4, TP8, TP4 x CP2 | `exact_norm_equal_all_ranks` and `exact_clipped_equal` true on all four; `exact_norm_values` one value, 10.974251850415355 = `one_gpu.exact_norm` |
| D head `b446400f` | 1591 passed / 1 failed / 153 skipped |
| D base `02b123e0` | 1555 passed / 9 failed / 153 skipped |

## A. One GPU

Timing (`timing.json`, medians of 30 runs, `<E2>`, run first on an idle GPU):

| Call | B200 torch / verl-omni (ms) | B200 RL-Kernel (ms) | H100 torch / verl-omni (ms) | H100 RL-Kernel (ms) |
|---|---|---|---|---|
| norm | 1.45 | 8.95 | 2.495 | 10.715 |
| norm + clip | 1.92 | 9.49 | 3.468 | 11.964 |
| transition, video, sampling | 0.39 | 0.27 | 0.668 | 0.475 |
| transition, video, scoring | 0.34 | 0.23 | 0.597 | 0.427 |
| transition, audio, sampling | 0.31 | 0.23 | 0.592 | 0.473 |
| transition, audio, scoring | 0.28 | 0.20 | 0.527 | 0.423 |

Norm partitions (`norm-partition.json`): 16 layouts, unpartitioned exact norm 10.974251850415355 (torch 10.974251747131348); exact 16 / 16 bitwise, torch 15 / 16.

Megatron hook (`megatron-hook.log`): the base `reproducible.py` (#730) falls back to `libnvrtc.so.12` and fails to load it, so the script stops before writing `megatron-hook.json`. NVRTC files in this environment (`A/nvrtc-files.txt`): only `libnvrtc.so.13` / `libnvrtc-builtins.so.13.0` (and the `.alt` variants), in the base environment's `nvidia/cu13/lib` and in the CUDA 13 toolkit; no CUDA 12 NVRTC; `LD_LIBRARY_PATH` unset.

## A2. #730 on Hopper

`row-mean-bytes.json`: `all_bytes_equal: true` against the B200 reference (capability [9, 0] here). `test-row-mean.txt`: 22 passed on 2 GPUs.

## B. verl-omni comparisons (`<E2>`)

`verl-omni-compare.json`: 560 cases, `rl_kernel_commit` `b446400f`, verl-omni `a634f4d8`; `prev_sample`, `prev_sample_mean`, `std_dev_t` and `sqrt_dt` 560 / 560, `log_prob` 206 / 560 with max abs difference 1.788e-07 (B200: 206 / 560, 1.8e-07). `logprob-batch.json`: `rl_kernel_differ` 0, `verl_omni_differ` 33 of 144 comparisons (B200: 27). `transition-tests.txt`: 9 passed.

## C. Multi-GPU norm (`<E1>`)

| Layout | Ranks | Exact norm equal on all ranks | Exact clipped equal | Exact norm | torch norm (all ranks) | torch clipped equal |
|---|---|---|---|---|---|---|
| TP2 | 2 | true | true | 10.974251850415355 | 10.974251747131348 | true |
| TP4 | 4 | true | true | 10.974251850415355 | 10.974251747131348 | true |
| TP8 | 8 (two nodes) | true | true | 10.974251850415355 | 10.974251747131348 | true |
| TP4 x CP2 | 8 (two nodes) | true | true | 10.974251850415355 | 10.974251747131348 | true |

The one-GPU values are the same (exact 10.974251850415355, torch 10.974251747131348), so the torch columns give the one-GPU norm and clipped bytes on all four layouts, as on B200.

## D. Test suites (`<E1>`, 2 + 2 GPUs on one node)

`_C` built in both checkouts (`build-*.log`, rc 0; `c-file-*.txt` inside each checkout).

- Head `b446400f`: 1591 passed, 1 failed, 153 skipped. Failure: `tests/ops/norm/test_rms_norm.py::test_registry_dispatches_rms_norm`.
- Base `02b123e0`: 1555 passed, 9 failed, 153 skipped. Failures: the same `test_registry_dispatches_rms_norm`, and 8 `tests/ops/norm/test_reproducible_norm.py` accelerator tests (`test_accelerator_bins_preserve_exact_sum_under_repartition`, `test_accelerator_device_context_is_restored`, `test_accelerator_warp_bins_with_mixed_exponents_and_nonfinite[1, 31, 32, 33, 257, 32771]`), all `libnvrtc.so.12: cannot open shared object file`.
- Nothing fails only at head. `test_h3_token_refiner.py::test_nodes_accuracy_repeat_batch_and_isolated[1-257]` (failing on B200) passes here. The skipped count differs from B200 (153 vs 148) and so the passed counts differ.

## Deviations

1. `<E1>` has no `torchrun` executable, so every `torchrun ...` in the README ran as `<E1>/bin/python -m torch.distributed.run ...` with the same arguments.
2. No `<OVERLAY>` was needed: `<E1>` already has accelerate 1.15.0.
3. `<E2>` was built new. verl's metadata pins `transformers<5.17`; the README asks for 5.19.0, so that pin was overridden at install time. Installing verl-omni's dependencies also brought in vllm 0.33.0 and vllm-omni (used by another task); nothing in this task imports them.
4. A run at the superseded heads (`164a5ebd` / `23ea31cc`) was started before the README update; it is not part of these results.
5. `megatron-hook.json` is missing because the script exits on the base side (see A).
