# B200 evidence: Qwen3-Next PR #468 (gated RMSNorm CUDA kernel), rerun at the current head

PR #468, fork branch `feat/cuda-qwen3-next-gated-rmsnorm`, pinned head `1e2eaec0` (full `1e2eaec06d3ce952f39e1027a60ee6e2e0f29904`), sm_100, labelled B200 only; measured with the same protocol as the H100 set `h100-pr468-h100`.

One NVIDIA B200, torch 2.13.0+cu130, vLLM 0.30.0, transformers 5.17.0, FlashInfer 0.6.18.post1; extension built in place from the pinned worktree, `rl_engine._C` verified to load from it (`env.txt`). Transformer Engine, Megatron-core, Liger and FLA are not installed, so those comparison rows are absent. Files: `tests.json`, `env.txt`, `logs/`, JSON reports and figures; `redaction.json` counts the placeholders.
