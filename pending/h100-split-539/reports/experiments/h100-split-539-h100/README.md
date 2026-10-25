# H100 evidence: split stack #539 (539a–c re-cut)

Measured on 4× H100 (sm90). Fork branches `feat/h3-one-block-replay` @ `193f079e` (539a), `feat/h3-full-model-qualification` @ `5ecb92de` (539b), `feat/h3-generation-gate` @ `6fe346df` (539c). Earlier working names `split/h3-539*` were deleted from the fork; the heads did not change.

## Commands

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EXTRA_PYTHONPATH=<HOME>/datastorage/sglang_probe/pkg
export RL_KERNEL_H3_WEIGHTS=<weights dir>
RL_KERNEL_H3_STRICT=1 python setup.py build_ext --inplace
python preimport_pytest.py tests/models/minimax_h3 tests/backends/cuda/test_h3_full_attention.py -q --tb=short -p no:cacheprovider -rsf
```

NCCL (539a only): `preimport_pytest.py ... -k nccl -v -rs` with `WORLD_SIZE` unset.

## Skips

8-GPU NCCL parametrizations not run (4-GPU node).

## Files

`tests.json`, `logs/*.log`, `redaction.json`, `SHA256SUMS`
