set -o pipefail
python3 - <<'PY'
import torch
assert torch.cuda.device_count() >= 4, "the real-model check needs four GPUs"
PY
log="$RUNNER_TEMP/real-model-pytest.log"
python3 -m pytest -q -rs tests/models/qwen3_next/check_qwen3_next_vllm_model_contract.py \
  tests/models/qwen3_next/check_qwen3_next_vllm_real_model.py | tee "$log"
if grep -Eq '^SKIPPED|[0-9]+ skipped' "$log"; then
  echo "::error::a real-model check was skipped"
  exit 1
fi
grep -Eq '[0-9]+ passed' "$log"
