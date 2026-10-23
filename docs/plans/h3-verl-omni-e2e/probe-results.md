# Stage A / B1 probe results (sanitized)

## VeRL-Omni versions (pinned in probe)

| Component | Revision |
| --- | --- |
| verl-omni | `efb9dc7f990cd9adc65e91add724f65266522a10` |
| vllm-omni | `12e9280010afc41ce5c6a1f46900fc329cf08377` |
| vllm | 0.30.0 |
| torch | 2.13.0+cu130 |

## CPU

- H3 pipeline + context-parallel CPU tests: **270 passed**.

## GPU — tiny FlowGRPO T2VA

- Topology: 2×H100, `--num-gpus 2`, `--actor-sp 2`, `--total-steps 2`.
- Result: **PASSED** (`EXIT_CODE=0`).
- Native P/P logprob drift (tiny random weights, step 1): `logprob_abs_diff_max ≈ 1.3e-3` (non-zero).

## GPU — LoRA TP=2 actor/rollout regression

- Script: `tests/special_e2e/run_minimax_h3_lora_sync_tp2.sh` (`NUM_GPUS=2`).
- Date: 2026-10-23.
- Result: **`MiniMax H3 LoRA TP=2 actor-to-rollout regression: PASS`** (rank 0/1 qkv/fc1/out_proj/fc2 LoRA parity, text_encoder TP=2 broadcast OK).

## Weights

- **Full transformer (RFC tests):** `hf_full/transformer` — **14 / 14** shards (~62 GiB index total size), revision `42ed227e`.
- **Conditioning-only tree:** `hf/transformer` — shard 1 only (~4.5 GiB) for `prepare_h3_weights.py` extract path.
- Use `export RL_KERNEL_H3_WEIGHTS=<weights root>`; `h3_weights.h3_shard_root()` prefers `hf_full/transformer`.

## Repro (GPU node)

```bash
# Tiny smoke (from verl-omni repo root, venv activated, 2 GPUs)
python tests/special_e2e/run_flowgrpo_minimax_h3_tiny.py --task t2va --num-gpus 2 --actor-sp 2 --total-steps 2

# TP=2 LoRA sync
bash tests/special_e2e/run_minimax_h3_lora_sync_tp2.sh
```
