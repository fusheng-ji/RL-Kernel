# H100 evidence: split stack #543 (543a–c)

Fork branches `split/qn-543a-foundation`, `543b`, `543c`.

## Commands (representative)

```bash
export CUDA_VISIBLE_DEVICES=0,1
pytest tests/models/qwen3_next tests/validation/common -q -p pytest_h100_megatron_plugin
# sm90 diagnostic (after FORCE_SM90 pip install -e .):
pytest tests/validation/common/test_logprob_comparison.py -k sm90_diagnostic -v
```

## Skips

One **ROCm 4-GPU** case per qn log (expected on CUDA H100).

## Files

- `tests.json` — summary
- `logs/*.log` — desensitized pytest stdout
- `redaction.json`, `SHA256SUMS`
