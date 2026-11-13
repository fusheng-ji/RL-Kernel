# H100 re-run for #548 at head `2202486d` (2026-11-12/13)

4 x NVIDIA H100 80GB (driver 580.173.02), torch 2.13.0+cu130, vLLM 0.30.0, `KERNEL_ALIGN_FORCE_SM90=1`, `TORCH_CUDA_ARCH_LIST=9.0`. Two nodes with 4 H100 each, no 8-GPU node.

| check | result |
|---|---|
| unit tests (5 PPO test files) | 44 passed, 1 skipped |
| megatron framework run (Actor TP=2, Critic TP=1, 2 PPO rounds) | PASS |
| deepspeed framework run (same layout) | PASS |

## What was run
The five PPO unit-test files and `tests.distributed.tp.check_tp_ppo_rollout` for `megatron` and `deepspeed` (dedicated GPU layout, vLLM TP=1 rollout, tiny-GPT configuration), as in the PR's earlier Hopper runs.

At the earlier head `f0b58af2` both framework rows failed in the harness (checkpoint conversion, `deepspeed` import). Both **pass at this head**. The Qwen3-8B qualification matrix was not run (no fixed-rollout thresholds in the tree; unchanged).

Raw logs are private; paths and host names are replaced (see `redaction.json`).
