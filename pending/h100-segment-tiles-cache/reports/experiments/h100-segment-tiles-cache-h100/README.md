# H100 report: segment-tiles cache and #588 resume check (2026-11-02)

Hardware: 4 x NVIDIA H100 80GB per node (driver 580.173.02, CUDA 13.0), two nodes, no single 8-GPU node.
torch 2.13.0+cu130, strict H3 build (`RL_KERNEL_H3_STRICT=1`), `KERNEL_ALIGN_USE_FAST_MATH=0`.
Paths below are placeholders (`<REPO>`, `<CKPT>`, `<OUT>`).

Trees: `tc-base` = `a88b5a09`; `tc-cache` = `a88b5a09` + patch (`b523938a`); `tc-588` = `67365734`;
`tc-588c` = `67365734` + patch (`fd41b902`). One `.so` built once and copied; `rl_engine._C` path checked in each tree.

## 1. Backward segment-tile cache

### A. Tests (2 GPUs visible, block-0 weights set)

| tree | result | skips |
|---|---|---|
| tc-cache | 395 passed, 2 skipped, 0 failed | `test_h3_sp_norm_adaln.py:288` needs 4 GPUs; needs 8 GPUs |
| tc-base | 389 passed, 2 skipped, 0 failed | same |

The 6 extra tests in tc-cache are the new ones; no test differs otherwise.

4-GPU re-run of the SP norm NCCL test (`test_nccl_region_byte_equal_to_ws1`, 4 GPUs visible):
world 2 PASSED, world 4 PASSED, world 8 SKIPPED (needs 8 GPUs), on both tc-cache and tc-base.
World 8 needs one node with 8 GPUs (CUDA-IPC fixed tree is an in-node mechanism); not run.

### B. Timings, one idle card (`nvidia-smi` 0 %), base/cache alternated 3 times, S=4097, H=5376 (ms, min / median)

| item | base | cache |
|---|---|---|
| `_segment_tiles` | 0.480 / 0.482 | 0.453 / 0.480 |
| gate_residual fwd+bwd | 1.013 / 1.016 | 0.444 / 0.447 |
| rmsnorm_mod fwd+bwd | 1.241 / 1.298 | 0.668 / 0.668 |

`probe_50_blocks.py small`: cached 1 tile build, uncached 200; `input grad bitwise equal cached vs uncached: True`,
cached repeat True; peak 23.7 GiB. (`nvidia-smi -q` saved as `nvidia_smi_q.txt`.)

### C. Real model (tc-588c, one H100, checkpoint `<CKPT>`, candidate mode)

| run | tile_builds | consumer_lookups |
|---|---|---|
| cached | 50 | 200 |
| `H3_TILES_NO_CACHE=1` | 200 | 200 |

Reading: the model hands every block a fresh `adaln_indices` tensor, so there is **no cross-block hit**;
only the within-block saving (4 builds -> 1) applies. The 638 parameter-gradient files of the two runs are
byte-identical (SHA-256, 0 differing). Step times: forward 1.91 s vs 1.84 s, backward 380.9 s vs 380.3 s,
total 513.5 s vs 513.6 s (the saving is not visible at model level).

## 2. #588 on H100

### A. Resume check (one H100, preflight size 128x128, 6 steps)
`resume-check.json`: 8 files compared (5 step files, `initial.pt`, `video.npy`, `audio.npy`).
- `cut` (cut after 2, 4, 5 calls): `differing` empty; final attempt `real_forward_calls` 0, 5 steps replayed.
- `cutverify` (cut after 2, `--verify-last-step`): `differing` empty; step 1 recomputed and verified.
- `trace_calls_match` true for both; the script's assertions passed.
Note: the first `cutverify` attempt hit CUDA out-of-memory because another process shared the card; it was re-run alone on an idle GPU (whole and cut unchanged).

### B. Preflight with memory/time readout (one H100)
128x128, 124 frames, 1 step: 101.9 s per step, 276.8 s total, `peak_allocated_bytes` 77,895,956,480 (72.5 GiB). Status `generation_complete_comparison_pending`.

### Extra: full size, one step, one H100
`generation-contract.json` size (1344x768, 124 frames), candidate, seed 42, `--stop-after-step 1` (stop at the end is intended).
No OOM. First sampling step about 1851 s (about 1.23x the B200 figure of about 1500 s, under the 1.5x stop rule).
`nvidia-smi` sampling peaked at 92,491 MiB of 93 GiB, which includes cached memory (the pipeline uses automatic CPU offload with a 20 GB reserve margin), so it shows the card is full, not the minimum requirement. About 50 steps would take about 25 h per seed, so the three-seed generation is not practical here.

## Not run
- 8-GPU single-node cases (item 3, and world 8 above): no node with 8 H100.
- #588 C (gate and three-seed generation): needs an H100 reference and rule not in the repository, about 60 GPU hours.

## Files
`logs/`: test logs, timing/probe log, `tile_builds_*.json`, `model_step_times_*.json`, `resume-check.json`, `preflight_128x128.json`, `fullsize_one_step.json`.
Raw logs are kept private; paths and host names are replaced (see `redaction.json`).
