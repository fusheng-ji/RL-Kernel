# B200 evidence: split stack #539 (539a-c)

Fork branches `feat/h3-one-block-replay` (539a), `feat/h3-full-model-qualification` (539b), `feat/h3-generation-gate` (539c): pytest on 2 x NVIDIA B200, plus one 8 x B200 NCCL run.

## Commands (representative)

```bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=.:<overlay with sglang 0.5.22 and matplotlib>
export RL_KERNEL_H3_WEIGHTS=<weights dir>
RL_KERNEL_H3_STRICT=1 python setup.py build_ext --inplace
pytest tests/models/minimax_h3 tests/backends/cuda/test_h3_full_attention.py -q
# NCCL byte-equality, WORLD_SIZE unset:
pytest tests/models/minimax_h3/test_h3_sp_norm_adaln.py tests/models/minimax_h3/test_h3_tp_adaln.py -k nccl -v
```

## Re-cut

The layers sit on the rewritten #494 head `4ef511c9` (heads `193f079e`, `5ecb92de`, `6fe346df`). The log headers show the local working names (`recut/h3-...`); the same heads are published as the `feat/h3-...` branches above. The 8-GPU NCCL log was measured on the heads before the re-cut (`cbdf48b0`, `44b7f832`); the layer code is identical, only nine docs files differ (commit-hash text).

## Skips

Four per layer in the 2-GPU logs: the 4- and 8-GPU NCCL cases. They were run on an 8-GPU node: `logs/539a-539c-nccl-2-4-8gpu-b200.log` (6 passed at each of 539a and 539c).

## Files

- `tests.json`: summary
- `logs/*.log`: desensitized pytest stdout
- `redaction.json`, `SHA256SUMS`
