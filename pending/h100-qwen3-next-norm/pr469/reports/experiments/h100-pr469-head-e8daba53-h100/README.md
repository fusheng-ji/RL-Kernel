# H100 evidence: Qwen3-Next PR #469 (C6 GDN decode-step goldens)

PR #469, fork branch `feat/ws1-c6-gdn-recurrent-golden`, pinned head `e8daba53591bdf0afb5ebd084a1e9c1f2eaacb85` (sm90, labelled H100 only).

Unit: `pytest tests/models/qwen3_next tests/ops/norm/test_rms_norm.py tests/ops/autograd/test_vjp_fp32.py tests/runtime/test_dispatch.py tests/validation/operators/test_ws1_gtest_gpu.py -q -p no:randomly -rfs`
Provider checks: `pytest -q -rs tests/models/qwen3_next/check_qwen3_next_norm_providers.py tests/models/qwen3_next/check_gdn_recurrent_golden.py`
Provider agreement: `tools/validation/models/ws1_gdn_provider_agreement.py`
