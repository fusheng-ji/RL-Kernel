# H100 evidence: Qwen3-Next norm PR #467

PR #467, fork branch `feat/cuda-qwen3-next-c1-norm`, pinned head `d42c8357` (sm90, labelled H100 only).

## Commands

Unit: `pytest tests/models/qwen3_next tests/ops/norm/test_rms_norm.py tests/ops/autograd/test_vjp_fp32.py tests/runtime/test_dispatch.py tests/validation/operators/test_ws1_gtest_gpu.py -q -p no:randomly -rfs`
Evidence: `tools/validation/models/qwen3_next_norm_evidence.py --out report.json`
Reuse: `tools/validation/models/qwen3_next_norm_reuse_check.py`

See `tests.json`, `logs/`, `report.json`, figures.
