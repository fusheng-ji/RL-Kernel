# B200 evidence: Qwen3-Next PR #469 (C6 GDN decode-step goldens), rerun at the current head

PR #469, fork branch `feat/ws1-c6-gdn-recurrent-golden`, pinned head `91bf5294` (full `91bf52943499b619e166886ac11dd432dd430440`), sm_100, labelled B200 only; measured with the same protocol as the H100 set `h100-pr469-h100`.

One NVIDIA B200, torch 2.13.0+cu130, vLLM 0.30.0, transformers 5.17.0, FlashInfer 0.6.18.post1; extension built in place from the pinned worktree, `rl_engine._C` verified to load from it (`env.txt`). Transformer Engine, Megatron-core, Liger and FLA are not installed, so those comparison rows are absent. Files: `tests.json`, `env.txt`, `logs/`, JSON reports and figures; `redaction.json` counts the placeholders.
