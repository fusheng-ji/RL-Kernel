# B200 evidence: split stack #543 (543a-c)

Fork branches `feat/ws1-qwen3-next-chain-foundation` (543a), `feat/ws1-qwen3-next-chain-training-l2` (543b), `feat/ws1-qwen3-next-chain-rollout-runners` (543c): pytest on 2 x NVIDIA B200.

## Commands (representative)

```bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=.:<overlay with megatron-core 0.16.1>
python setup.py build_ext --inplace
python <pre-import wrapper> tests/models/qwen3_next tests/validation/common -q   # imports megatron.core, then runs pytest
```

## Skips

Two per layer: the SM90 diagnostic test (needs Hopper; it passes in the H100 evidence) and the distributed logprob test that needs four ROCm GPUs. The four megatron-dependent tests of 543b and 543c run here (megatron-core installed).

## Files

- `tests.json`: summary
- `logs/*.log`: desensitized pytest stdout
- `redaction.json`, `SHA256SUMS`
