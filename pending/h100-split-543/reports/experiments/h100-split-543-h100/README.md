# H100 evidence: split stack #543 (543a–c)

Fork branches `feat/ws1-qwen3-next-chain-foundation` @ `1ee46a34` (543a), `feat/ws1-qwen3-next-chain-training-l2` @ `5778bc88` (543b), `feat/ws1-qwen3-next-chain-rollout-runners` @ `ac902598` (543c). Earlier working names `split/qn-543*` were deleted from the fork; the heads did not change.

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
