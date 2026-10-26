#!/bin/bash
# PR #468 on an idle B200 at its head: unit suite, check_operator, C3/C4 gates, norm evidence and the two reuse checks.
source "$(dirname "$0")/../../b200-gaps-common/common.sh"
export TMPDIR=/tmp/q468; mkdir -p "$TMPDIR"   # short node-local path: a long TMPDIR breaks vLLM's distributed init
export PYTHONPATH=$WT
env_header
run pytest $PY -m pytest tests/models/qwen3_next tests/ops/norm/test_rms_norm.py tests/ops/autograd/test_vjp_fp32.py tests/runtime/test_dispatch.py tests/validation/operators/test_ws1_gtest_gpu.py -q -p no:randomly -rfs
run check_operator_rms_norm $PY tools/validation/operators/check_operator.py --op rms_norm --candidate cuda --device cuda --dtype bf16 --batch 2 --seq 16 --normalized-dim 4096 --seed 123 --check-grad
run check_operator_rms_norm_gated $PY tools/validation/operators/check_operator.py --op rms_norm_gated --candidate cuda --device cuda --dtype bf16 --batch 2 --seq 16 --head-dim 128 --seed 123 --check-grad
run check_operator_qwen3_next_rms_norm $PY tools/validation/operators/check_operator.py --op qwen3_next_rms_norm --candidate cuda --device cuda --dtype bf16 --batch 2 --seq 16 --normalized-dim 4096 --seed 123 --check-grad
M=rl_engine/validation/models/qwen3_next_norm_manifest.json
for op in qwen3_next_rms_norm rms_norm_gated; do
  run c3c4_forward_$op $PY tools/validation/operators/check_forward_invariance.py --manifest $M --op $op --candidate cuda --backend-profile cuda_bf16 --hidden 2048 --head-dim 128
  run c3c4_gradient_$op $PY tools/validation/operators/check_gradient_invariance.py --manifest $M --op $op --candidate cuda --backend-profile cuda_bf16 --hidden 2048 --head-dim 128
done
run norm_evidence $PY tools/validation/models/qwen3_next_norm_evidence.py --out "$OUT/report.json"
run reuse_qwen3 $PY tools/validation/models/qwen3_next_norm_reuse_check.py --op qwen3_next_rms_norm --out "$OUT/qwen3_next_rms_norm_reuse.json"
run reuse_gated $PY tools/validation/models/qwen3_next_norm_reuse_check.py --op rms_norm_gated --out "$OUT/rms_norm_gated_reuse.json"
finish
