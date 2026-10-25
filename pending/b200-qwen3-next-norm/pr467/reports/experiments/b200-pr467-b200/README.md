# B200 evidence: Qwen3-Next PR #467 (zero-centred RMSNorm (C1)), rerun at the current head

PR #467, fork branch `feat/cuda-qwen3-next-c1-norm`, pinned head `d42c8357` (full `d42c83571c2fd43ce02d62ec32dcefb8a060d429`), sm_100, labelled B200 only; measured with the same protocol as the H100 set `h100-pr467-h100`.

One NVIDIA B200, torch 2.13.0+cu130, vLLM 0.30.0, transformers 5.17.0, FlashInfer 0.6.18.post1; extension built in place from the pinned worktree, `rl_engine._C` verified to load from it (`env.txt`). Transformer Engine, Megatron-core, Liger and FLA are not installed, so those comparison rows are absent. Files: `tests.json`, `env.txt`, `logs/`, JSON reports and figures; `redaction.json` counts the placeholders.
