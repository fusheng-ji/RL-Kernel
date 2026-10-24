# H100 evidence: split stack #539 (539a–c)

Fork branches `split/h3-539a-one-block`, `539b`, `539c` — Hopper pytest on `<HOME>/datastorage` worktrees.

## Commands (representative)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EXTRA_PYTHONPATH=<HOME>/datastorage/sglang_probe/pkg
export RL_KERNEL_H3_WEIGHTS=<HOME>/datastorage/models/h3
pytest tests/models/minimax_h3 tests/backends/cuda/test_h3_full_attention.py -q
```

## Skips

Two **8-GPU NCCL** cases per main log; **not run** (4-GPU Hopper nodes only). See `logs/539a-nccl-4-8gpu.log` for supplemental 4-GPU NCCL pass.

## Files

- `tests.json` — summary
- `logs/*.log` — desensitized pytest stdout
- `redaction.json`, `SHA256SUMS`
