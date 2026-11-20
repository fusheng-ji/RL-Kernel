# H100 round, GDN one-launch recurrence: `local/e2e-bench-gdn-fused` @ `213a9bcd` (2026-11-20)

Results for the README of `handoff/h100-qwen3-next-gdn-fused-20261120` (bundle refs `perf/qwen3-next-gdn-fused-recurrence` `63d46013`, `local/e2e-bench-gdn-fused` `213a9bcd`; everything ran from `213a9bcd`). Hardware: 4 x NVIDIA H100 per node (95830 MiB each, NVLink NV6, `nvidia-smi.txt`), driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, vLLM 0.30.0, FlashInfer 0.6.18.post1, transformer_engine 2.18.0, megatron-core 0.16.0rc0, apache-tvm-ffi 0.1.11, **flash-linear-attention 0.4.2 and fla-core 0.4.2** (installed with `pip install --no-deps` into a *copy* of the training environment made for this round; the shared environment was not touched; `pip-list-filtered.txt`), Python 3.12.15. Detached worktree at `213a9bcdfdf3822e9bd8b5ef646078efdcc56ff6`, `core.fileMode=false`, `git status --porcelain --untracked-files=no` empty before and after every step. `PYTHONNOUSERSITE=1`, `CUDA_DEVICE_MAX_CONNECTIONS` unset, `CUDNN_HOME` and `LD_LIBRARY_PATH` at the base environment's nvidia directories, `TRITON_F32_DEFAULT=ieee` for A. `rl_engine._C.__file__` is inside the worktree (`A/c-file.txt`; chain stages build their own in the run clone).

The follow-up handoff (`8284ce2b`) was run first in the same environment (separate branch `handoff/h100-qwen3-next-followup-results-20261119`).

## One row per step
| Step | Result |
|---|---|
| A bits and per-layer time (1 GPU) | **no `false`**: all four probe lines have `output_equal` and `state_equal` true and `output_max_abs` 0.0 (table below). `checks.log`: **43 passed, rc 0**. |
| B TP4 chain + `bi` full | **passed**: `preflight passed, layers passed, l2 passed, bi passed`; every run `source_commit` `213a9bcd`; `bi-matrix.json` 97 cells, all passed; chain report `passed=True first_drift=None cells=16`; gate rc 0. |
| C EP4 chain | **preflight passed, layers passed, l2 passed** (rerun; the first attempt, with a foreign job on the node, is in `not-green-attempts/`). |
| D end-to-end speed test | `rc 0`, `rl_kernel_commit` `213a9bcd`, `tracked_tree_dirty` false. Rollout: the `native_bi` arm is in `unsupported` ("VLLM batch_invariant mode is not supported for GDN_ATTN." for certified and long), as expected. Strict rollout checks: none false. **Replay has failures**: the stock arm `native` (VIME's own Qwen3-Next model) failed in every round, see below. Strict replay is measured. |
| E CPU suite | **15 failed, 658 passed, 33 warnings**; all 15 failures are the known `AssertionError: Invalid device id` of `test_qwen3_next_training_checkpoint.py`. |

### A: probe, one H100 (B200 reference in the README: 0.72 / 2.68 / 4.10 / 1.73 ms vs 23.5 / 380.6 / 583.7 / 154.2 ms)
| lengths | cache slots | output / state equal | output_max_abs | one launch (ms) | decode loop (ms) |
|---|---|---|---|---|---|
| 64 | 65 | yes / yes | 0.0 | 1.06 | 29.7 |
| 1024 | 65 | yes / yes | 0.0 | 4.58 | 470.3 |
| 1024, 777, 1500, 64 | 65 | yes / yes | 0.0 | 6.40 | 692.7 |
| 512 (x16) | 362 | yes / yes | 0.0 | 2.89 | 265.3 |

`probe.jsonl` also holds three non-JSON log lines the script wrote to stdout (kept as produced).

### D: strict arm, prefill seconds per shape (pre-fix B200 reference in the README: strict prefill 0.89 / 3.0 / 11.6 s for 64 / 256 / 1024-token prompts, replay 1.0 / 2.8 / 9.6 s)
- strict rollout certified prefill: p64-b1-r16: 0.29 s, p64-b4-r16: 0.59 s, p256-b1-r16: 0.31 s, p256-b4-r16: 0.64 s, p1024-b1-r16: 0.46 s, p1024-b4-r16: 1.02 s
- strict rollout long prefill: p512-b1-r512: 0.38 s, p512-b16-r512: 1.14 s, p512-b64-r512: 2.99 s
- strict replay forward_logprob (median): certified p64-b1-r16: 0.29 s, p64-b4-r16: 0.31 s, p256-b1-r16: 0.29 s, p256-b4-r16: 0.32 s, p1024-b1-r16: 0.42 s, p1024-b4-r16: 0.57 s; long p512-b1-r512: 0.42 s, p512-b4-r512: 0.58 s, p512-b16-r512: 1.45 s
- `strict_over_native` in the report is the ratio of the strict arm to the stock arm (the report's own field, I did not recompute it), e.g. rollout `native_graph` certified: p64-b1-r16: 45.6x, p64-b4-r16: 21.6x, p256-b1-r16: 25.0x, p256-b4-r16: 16.8x, p1024-b1-r16: 27.0x, p1024-b4-r16: 18.1x; replay has none (its stock arm failed)

(`bench/report.json` has every arm, shape, round and per-repeat value; `strict_over_native` has the strict-to-stock ratios.)

### D: replay failure of the stock arm
`replay.failures` (excerpt): [{"arm": "native", "workload": "certified", "errors": ["worker exited with 1; see logs/replay-native-certified-round0.log", "worker exited with 1; see logs/replay-native-certified-round1.log"]}, {"arm": "native", "workload": "long", "errors": ["worker exited with 1; see logs/replay-native-long-round0.log", "worker exited with 1; see logs/replay-native-long-round1.log"]}]
Cause in `bench/logs/replay-native-certified-round0.log`: `ValueError: Shape mismatch loading module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0: HuggingFace (1024, 2048), Megatron (256, 2048)` (VIME's Megatron model loading the expert weights under TP4). It is not the `NameError: Qwen3NextAttention` from the missing `flash-linear-attention`, which was installed. Sent as found.

## Not-green attempt (kept unedited, `not-green-attempts/`)
- **C first attempt**: `preflight` passed; `layers` failed with `gpu_memory_not_released` (a process holding 37 GB). Its `header.txt` process lines and the `nvidia_smi` field of its preflight `attempts/0/environment.json` show two other processes (about 20 GB each) on the node's GPUs at the start, from another job of the same user that I did not start, so this is not a clean result; the rerun on an idle node passed.

## Notes
- B and D ran on one node, the first C attempt on the other; the follow-up round used both nodes too. B's report and gate were run with `PY=<the copy's python>` as asked.
- `A/probe.log` is empty of errors; times are UTC; node and allocation names are redacted.
