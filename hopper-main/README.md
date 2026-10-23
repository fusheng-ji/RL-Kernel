# main-branch PRs on Hopper (H100)

torch 2.13.0+cu130, 2 to 4 x NVIDIA H100, each PR at its own head. `pr491` (deterministic all-gather before/after, 2 and 4 GPUs; no 8-GPU data), `pr536` (HAL auto-tuning correctness and 52-case benchmark; `performance_summary.json` drops the per-trial device identities), `pr548` (TP PPO: unit tests and the Megatron / DeepSpeed two-round runs on a tiny GPT; the Qwen3-8B matrix was not run).
