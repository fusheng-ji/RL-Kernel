# H3 on Hopper with strict boundaries (H100, 2026-11-22)

Handoff: `handoff/h100-h3-hopper-strict-20261122` (`fe04e2e1`), README sections S1 and S2.
Hardware: NVIDIA H100 80GB HBM3, 4 per node; S2 on one exclusive node (`node-a`).

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
