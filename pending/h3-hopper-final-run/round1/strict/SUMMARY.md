# H3 on Hopper with strict boundaries (H100, 2026-11-22)

Handoff: `handoff/h100-h3-hopper-strict-20261122` (`fe04e2e1`), README sections S1 and S2.
Hardware: NVIDIA H100 80GB HBM3, 4 per node. S2 on one exclusive node (`node-a`); S1's 2- and 4-rank runs and suites on `node-a`, its 8-rank runs on `node-a` + `node-b` (global rank 0 on `node-a`).

## S2: omni rollout with strict boundaries (bundle `5f1a750b` = #735 `4a4fe0c0` + #734 `dcc8535a`)

Environment `<E2>`: torch 2.14.0+cu130, triton 3.8.0, diffusers 0.43.0, transformers 5.19.0, peft 0.22.0,
accelerate 1.16.0, verl 0.10.0.dev0, verl-omni 0.4.0 (editable), vllm 0.33.0, vllm-omni 0.33.0rc2.dev206+g6039984e3.
All 49 entries of the B200 `pip-list-e2-b200.txt` match (`S2/pip-list-e2-*.txt`, `pip-list-filtered-E2.txt`).
`rl_engine._C` was built in the checkout (`S2/c-file.txt`, `S2/build.log`, rc 0); the tree stayed clean (`S2/status-*.txt`).

Verdict: **every S2 expectation is met** for both B random and B = 0. One deviation in the GPU test command (below).

| Check | B random | B = 0 |
|---|---|---|
| `engine.json` `pass` | true | true |
| `install[*].strict_boundaries` (4 ranks) | true ×4 | true ×4 |
| `ranks_agree` | true | true |
| batched calls with / without an alone twin | 4 / 0 | 4 / 0 |
| `captures` (`in_run`) | 2 (true, true) | 2 (true, true) |
| `record_errors` | none | none |
| `lora.pass` (tampered sync on rank 2, block 17 `attn.to_v.A[3, 5]`, refused on every rank, decoy kept, senders' digests) | true, 650 shards per rank | true, 650 shards per rank |
| `canonical.json` `pass` (video and audio bitwise, `in_rollout_digests`) | true | true |
| `actor.json` `pass` (rows per call, `differ`) | true (3957, 3957; empty) | true (3957, 3957; empty) |

Memory per rank (all 4 ranks identical, both runs): `resident_param_bytes` 37,379,054,144 and
`engine_parameter_bytes_released` 16,570,990,400, the same values as B200. vLLM-Omni reports
"Process-scoped GPU memory after model loading: 35.50 GiB" per worker (`engine.log`).

Timings (`engine.json` `seconds`):

| | B200 | H100 B random | H100 B = 0 |
|---|---|---|---|
| A alone | about 9 s | 15.33 s | 8.51 s |
| B alone | 3.9 s | 4.78 s | 4.79 s |
| A + B together | 6.8 s | 9.24 s | 9.19 s |

Tests:

- `omni-cpu.txt`: 38 passed, 3 skipped (CUDA required), as expected.
- `omni-gpu.txt`: **41 passed**: every test ran.

### Deviations

1. **GPU test weights path.** The README command `RL_KERNEL_H3_WEIGHTS=<CKPT>` ran with 38 passed and 3 skipped
   ("released H3 shards ($RL_KERNEL_H3_WEIGHTS) required"). `h3_shard_root()` in
   `rl_engine/validation/models/h3_weights.py` looks for `$RL_KERNEL_H3_WEIGHTS/{hf_full,hf}/transformer/`, so
   it needs the directory that holds `hf_full/`, not the checkpoint itself. I reran only that command with
   `RL_KERNEL_H3_WEIGHTS=<parent of CKPT>` on the same node; nothing else changed. That attempt is kept unedited in
   `S2/not-green-attempts/omni-gpu-weights-path-ckpt/`.
2. **Checkpoint layout.** The local `hf_full` (revision 42ed227e) had only `FL2VA/audio_vae`. `<CKPT>` is a
   directory of symlinks to every `hf_full` entry except `FL2VA`, plus a full download of `FL2VA` at the same revision.
3. **`<E2>` build.** verl's metadata pins `transformers<5.17`; the README pins 5.19.0, so the pin was overridden.
   Installing the verl-omni dependencies already brought in vllm 0.33.0 and vllm-omni `6039984e`, so the S2 install
   step only reinstalled vllm-omni at the same commit (`pip-list-e2-before.txt` and `-after.txt` differ only in its
   local version suffix).
4. `env.txt` was rewritten after the run with the same interpreter, because my recording helper had passed the venv
   directory instead of its python.

## S1: full model, strict boundaries (#732 `6c7e424c`, #733 `4e67d5d7`)

Environment `<E1>`: torch 2.13.0+cu130, triton 3.7.1, accelerate 1.15.0 (`pip-list-filtered-E1.txt`), with the H3 provider
(Diffusers with MiniMax-H3) at `<PROVIDER>/src` on `PYTHONPATH`; no `<OVERLAY>` needed. `<CKPT>` as in S2.
`_C` built in each checkout with `RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1` (`build-732.log`, `build-733.log`, rc 0;
`c-file-*.txt` inside each checkout); every tree stayed clean (`status-*.txt`).

Launch: every run is an `srun` step with one task per GPU, running `h3_parallel_full_model.py` directly (the first form the README allows),
`MASTER_ADDR` = `node-a`, a new `MASTER_PORT` per run (`*/command.txt`, `run_s1.sh.txt`). Flags are exactly the README tables, with no
`--provider-boundaries`. The runs are independent; only the order differs (all 1-node runs first, then all 8-rank runs).

Verdict: **every run meets the S1 expectation.** All 19 non-inject runs (7 of #732, 12 of #733) are bitwise with strict boundaries: `all_bitwise: true`,
every check true on every rank (for #733 including every LoRA gradient), `shards_missing` empty, `strict_boundaries: true`.
All four negative controls are caught where expected.

| Run | Ranks | `strict_boundaries` | Result | Peak / resident GiB per rank (`parallel_peak_bytes` / `resident_param_bytes`, max over ranks) |
|---|---|---|---|---|
| `harness_tp4_sp` | 4 (1 node) | true | **bitwise** | not recorded (full weights) |
| `sharded_tp2_40` | 2 (1 node) | true | **bitwise** | 45.5 / 43.8 |
| `sharded_tp4_sp_928` | 4 (1 node) | true | **bitwise** | 37.8 / 34.8 |
| `sharded_tp4_cp2_ring_925` | 8 (2 nodes) | true | **bitwise** | 36.7 / 34.8 |
| `sharded_tp4_cp2_sp_ring_928` | 8 (2 nodes) | true | **bitwise** | 37.3 / 34.8 |
| `sharded_tp4_cp2_sp_ulysses_928` | 8 (2 nodes) | true | **bitwise** | 37.3 / 34.8 |
| `sharded_tp8_sp_928` | 8 (2 nodes) | true | **bitwise** | 33.1 / 30.3 |
| `inject_sharded_tp4_sp_928` | 4 (1 node) | true | not bitwise (expected); `first_drift_forward` on all 4 ranks: block.17, row 700 (video), 1 row, owners [2] | 37.8 / 34.8 |
| `inject_sharded_tp4_cp2_ring_925` | 8 (2 nodes) | true | not bitwise (expected); `first_drift_forward` on all 8 ranks: block.17, row 5 (text), 1 row, owners [0, 1, 2, 3] | 36.7 / 34.8 |
| `lora_random_tp2_40` | 2 (1 node) | true | **bitwise** | 45.9 / 43.8 |
| `lora_random_tp4_sp_928` | 4 (1 node) | true | **bitwise** | 37.1 / 34.8 |
| `lora_random_tp4_cp2_ring_925` | 8 (2 nodes) | true | **bitwise** | 37.0 / 34.8 |
| `lora_random_tp4_cp2_sp_ring_928` | 8 (2 nodes) | true | **bitwise** | 37.1 / 34.8 |
| `lora_random_tp4_cp2_sp_ulysses_928` | 8 (2 nodes) | true | **bitwise** | 37.1 / 34.8 |
| `lora_random_tp8_sp_928` | 8 (2 nodes) | true | **bitwise** | 32.3 / 30.3 |
| `lora_zero_tp2_40` | 2 (1 node) | true | **bitwise** | 45.9 / 43.8 |
| `lora_zero_tp4_sp_928` | 4 (1 node) | true | **bitwise** | 37.1 / 34.8 |
| `lora_zero_tp4_cp2_ring_925` | 8 (2 nodes) | true | **bitwise** | 37.0 / 34.8 |
| `lora_zero_tp4_cp2_sp_ring_928` | 8 (2 nodes) | true | **bitwise** | 37.1 / 34.8 |
| `lora_zero_tp4_cp2_sp_ulysses_928` | 8 (2 nodes) | true | **bitwise** | 37.1 / 34.8 |
| `lora_zero_tp8_sp_928` | 8 (2 nodes) | true | **bitwise** | 32.3 / 30.3 |
| `inject_lora_tp4_sp_928` | 4 (1 node) | true | not bitwise (expected); `first_drift_forward` on all 4 ranks: block.17, row 700 (video), 1 row, owners [2]; `lora_gradients_bitwise` false on every rank | 37.1 / 34.8 |
| `inject_lora_tp4_cp2_ring_925` | 8 (2 nodes) | true | not bitwise (expected); `first_drift_forward` on all 8 ranks: block.17, row 5 (text), 1 row, owners [0, 1, 2, 3]; `lora_gradients_bitwise` false on every rank | 37.0 / 34.8 |

On the inject runs `first_drift_backward` is `block.49.grad` (audio rows) on every rank, as the injected forward change propagates through the backward pass.

### S1.c suites (`<E1>`, 2 + 2 GPUs, the README command)

- #733 (`suite-733.txt`): 1440 passed, 2 failed, 156 skipped.
- #732 (`suite-732.txt`): 1416 passed, 2 failed, 144 skipped.
- The same 2 failures on both sides: `tests/models/minimax_h3/test_h3_ring_attention.py::test_block_with_h3_attention_equals_single_gpu_bitwise[cp2_ring_len261]` and `[cp2_ulysses_h3]`.
  Their spawned rank raises `FileNotFoundError: set RL_KERNEL_H3_WEIGHTS to the prepare_h3_weights.py output dir`: the README's suite command
  does not set `RL_KERNEL_H3_WEIGHTS`, and these two multi-process tests fail instead of skipping when it is unset. Not rerun.
- **Nothing fails only at #733.** The B200 `test_h3_benchmark_timing.py::...[gather]` CPU failure does not occur here.
