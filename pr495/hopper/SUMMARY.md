# PR #495 H100 summary

head: `e612412c815d02d4d05dac1c5fe44d4688dcc82c`

## pytest (default build, 2 GPUs)

| passed | failed | skipped |
| --- | --- | --- |
| 371 | 0 | 34 |

Skip breakdown:
- 15× `batch_invariant_logp_sm90` (supplement `pytest_sm90_only.log`: still skipped — `build_ext` alone did not compile sm90 symbols; upstream expects `KERNEL_ALIGN_FORCE_SM90=1 pip install -e .`)
- 18× Ascend NPU kernels (not applicable on H100)
- 1× CUDA dispatch may select fused logp

## benchmark `report.json`

32768×151936 BF16: fused kernel forward+backward **38.1 ms**, peak **9.27 GiB** (94 GB H100; large-N memory test ran).

## vs B200 (`pr495/docs/.../fused-logp-backward-b200/report.json` on pr_assets)

Compare timing table in redacted `report.json`; do not reuse B200 counts verbatim.
