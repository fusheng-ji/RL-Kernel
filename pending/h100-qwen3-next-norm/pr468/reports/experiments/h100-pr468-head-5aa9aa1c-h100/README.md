# H100 evidence: Qwen3-Next norm PR #468

PR #468, fork branch `feat/cuda-qwen3-next-gated-rmsnorm`, pinned head `5aa9aa1cf446c11bd4374c57c9ab81e0fb7b59a7` (sm90, labelled H100 only).

Unit: `pytest tests/models/qwen3_next tests/ops/norm/test_rms_norm.py tests/ops/autograd/test_vjp_fp32.py tests/runtime/test_dispatch.py tests/validation/operators/test_ws1_gtest_gpu.py -q -p no:randomly -rfs`
Evidence: `tools/validation/models/qwen3_next_norm_evidence.py --out report.json`
Reuse: `tools/validation/models/qwen3_next_norm_reuse_check.py`
Operator gates and C3/C4 use the commands in the PR Tests section.
