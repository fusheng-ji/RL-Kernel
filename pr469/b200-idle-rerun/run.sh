#!/bin/bash
# PR #469 on an idle B200 at its head: unit suite, provider checks and the provider-agreement runner.
source "$(dirname "$0")/../../b200-gaps-common/common.sh"
export TMPDIR=/tmp/q469; mkdir -p "$TMPDIR"   # short node-local path: a long TMPDIR breaks vLLM's distributed init
export PYTHONPATH=$WT
env_header
run pytest $PY -m pytest tests/models/qwen3_next tests/ops/norm/test_rms_norm.py tests/ops/autograd/test_vjp_fp32.py tests/runtime/test_dispatch.py tests/validation/operators/test_ws1_gtest_gpu.py -q -p no:randomly -rfs
run check_providers $PY -m pytest -q -rs tests/models/qwen3_next/check_qwen3_next_norm_providers.py tests/models/qwen3_next/check_gdn_recurrent_golden.py
$PY tools/validation/models/ws1_gdn_provider_agreement.py > "$OUT/gdn_provider_agreement.json" 2> "$OUT/gdn_provider_agreement.log"
printf '%s\t%s\n' gdn_provider_agreement "$?" >> "$OUT/exitcodes.tsv"
finish
