# H100 evidence: Qwen3-Next PR #469 (C6 GDN decode-step goldens)

PR #469, fork branch `feat/ws1-c6-gdn-recurrent-golden`, pinned head `d1129d9f0e0d053d30f05649c4e7cd6ee5ee9604` (sm90, labelled H100 only).

Unit suite (step A) was measured at `836c483d`; step F (`ws1_gdn_provider_agreement.py`) failed there because the runner still imported `tests/check_gdn_recurrent_golden.py`. Re-run at `d1129d9f` (one-line path fix to `tests/models/qwen3_next/check_gdn_recurrent_golden.py`).

Provider: `tools/validation/models/ws1_gdn_provider_agreement.py` → `gdn_provider_agreement.json`.
