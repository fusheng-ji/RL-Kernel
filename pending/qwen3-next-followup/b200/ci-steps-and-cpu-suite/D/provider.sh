set -o pipefail
log="$RUNNER_TEMP/provider-pytest.log"
python3 -m pytest -q -rs tests/models/qwen3_next/check_qwen3_next_norm_providers.py tests/models/qwen3_next/check_gdn_recurrent_golden.py \
  tests/models/qwen3_next/check_qwen3_next_forward.py tests/models/qwen3_next/check_qwen3_next_attention.py \
  tests/models/qwen3_next/check_qwen3_next_gdn_bridge.py tests/models/qwen3_next/check_qwen3_next_gdn_sequence.py \
  tests/models/qwen3_next/check_qwen3_next_conv_bridge.py tests/models/qwen3_next/check_qwen3_next_core_matrix.py \
  tests/models/qwen3_next/check_qwen3_next_shared_core.py tests/models/qwen3_next/check_qwen3_next_paged_attention.py \
  tests/models/qwen3_next/check_qwen3_next_batch_invariance.py tests/models/qwen3_next/check_qwen3_next_bi_requests.py \
  tests/models/qwen3_next/check_qwen3_next_vllm_attention_metadata.py \
  tests/models/qwen3_next/check_qwen3_next_native_rollout.py tests/models/qwen3_next/check_qwen3_next_engine_metadata.py | tee "$log"
# Fail closed: these modules skip as a whole when CUDA or vLLM is missing, and pytest
# exits 0 for an all-skipped run. A skipped provider check is not provider evidence.
if grep -Eq '^SKIPPED|[0-9]+ skipped' "$log"; then
  echo "::error::a provider check was skipped"
  exit 1
fi
grep -Eq '[0-9]+ passed' "$log"
