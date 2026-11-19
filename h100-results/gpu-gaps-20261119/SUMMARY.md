# GPU gaps on H100 (2026-11-19 handoff, head `f0572969`)

Hardware: NVIDIA H100 (cc 9.0), 95830 MiB per card, driver 580.173.02; `gpu.txt` in each item folder. Versions in
`versions.txt` (torch 2.13.0+cu130 everywhere). Nothing in any branch was changed. All five worktrees were clean:
`status-*.txt` are empty and `diff-summary-*.txt` are empty (no mode changes in these checkouts).

| Item | Result | As expected? |
|---|---|---|
| G1 FL2VA, with providers | 43 passed | yes (B200: 43 passed) |
| G1 FL2VA, RL-Kernel env | 36 passed, 7 skipped | yes (B200: 36 / 7) |
| G1 Ref2VA, with providers | 40 passed | yes (B200: 40 passed) |
| G1 Ref2VA, RL-Kernel env | 36 passed, 4 skipped | yes (B200: 36 / 4) |
| G2 `tests/distributed`, `10552d82` | 7 failed, 237 passed, 31 skipped | see below |
| G2 `tests/distributed`, `ba811b90` | 3 failed, 241 passed, 31 skipped | see below |
| G2 `tests/models/minimax_h3`, `10552d82` | 1161 passed, 70 skipped, 0 failed | |
| G2 `tests/models/minimax_h3`, `ba811b90` | 1167 passed, 70 skipped, 0 failed | +6 passed |
| G3 `check_sglang_report.py` (TP1, TP2, TP4) | `RESULT: PASS` | yes, for the TP sizes run |
| G3 pytest | 54 passed, 3 skipped | yes (see below) |

## G1

`test_initial_rows_on_cuda_equal_cpu_bytes` is among the passed tests in every run (the "with providers" runs have
0 skips). Both environments are described in `versions.txt`; the providers environment is a virtualenv over the
RL-Kernel one with Diffusers 0.43.0, SGLang 0.5.21 (`sglang-kernel` 0.4.7), accelerate and torchaudio. It also needed
`addict`, `imageio` and `cache_dit`, which SGLang's multimodal package imports; and the tiny test pipeline needs Hub
access (it was cached there). The two earlier attempts, before those packages and the Hub access were in place, are in
`earlier-attempts/G1-first` and `G1-second` (5 skips; then 4 and 1 failures from the missing processor). They are not
results.

## G2

- Strict build in both worktrees: `env-*.txt` shows capability (9, 0) and `_C.__file__` inside each worktree.
- `diff.txt`, `distributed`: failing only on `ba811b90`: empty. Failing only on `10552d82`: exactly four:
  `test_run_world_reaps_workers[success]`, `[error]`, `[crash]` and `test_run_world_returns_rank_ordered_results`
  (the last is in `tests/distributed/sp/test_h3_sp_norm_adaln.py`).
- Failing on both trees (3, as on B200):
  `tests/distributed/cp/test_rocm_attention_transport.py::test_rccl_adapter_world_sizes_match_the_shared_collective`,
  `tests/distributed/tp/test_distributed_grpo_loss.py::TestSingleRank::test_dispatch_resolves_this_backend`,
  `tests/distributed/tp/test_h3_tp_adaln.py::test_nccl_byte_equal_to_ws1[2]`. The run used `--tb=no`, so the
  failure reasons are not recorded.
- `minimax_h3`: no failure on either tree, so both "failing only on" lists are empty, and `ba811b90` has 6 more passed
  tests than `10552d82`. The B200 failure `test_h3_token_refiner.py::test_nodes_accuracy_repeat_batch_and_isolated[1-257]`
  passes on H100, as predicted.
- The two trees ran at the same time, on different nodes and cards (two cards each).
- `RL_KERNEL_H3_WEIGHTS` pointed to a directory with the pinned `h3_block0.safetensors` and
  `h3_conditioning.safetensors` and `hf` linked to the released checkpoint at `42ed227e`.

## G3

- `check.txt`: at TP1, TP2 and TP4 every bound output equals the training operator bitwise on every rank and equals TP1,
  and is restored on `remove()`; readbacks are clean; plugin checks pass. The native table is informational. H100 values
  are the same as B200's in all rows except `final_out`: 0.0234 / 0.0234 / 0.0078 at TP1 / 2 / 4 on H100, against
  0.0078 / 0.0234 / 0.0234 on B200.
- **TP8 was not run**: only one 4-card node was fully free (the other node had a card occupied by another job). So
  `report.json` has TP1, TP2 and TP4.
- pytest: 54 passed, 3 skipped. The skips are the 8-GPU test (`needs 8 GPUs`) and two tests that need `vllm_omni`
  (not installed). On 8 x B200 it was 55 passed, 2 skipped, so the difference is that the 8-GPU test did not run.
- **Deviation:** the handoff says no RL-Kernel extension build is needed, but the SGLang tests and the report fail
  without `_C` (`the RL-Kernel CUDA extension is required`). I built `_C` in the `8506f6a2` worktree with a plain
  `setup.py build_ext --inplace` (`G3/build.log`) and ran again. The first attempt (10 pytest failures, no report) is in
  `earlier-attempts/G3-first`.

## Notes

- Host names, paths and the job id are replaced with `{...}` placeholders; hashes are untouched. No `*.safetensors`.
