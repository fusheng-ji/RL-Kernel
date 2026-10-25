# PR #495 H100 summary

head: `f30fc21ea9e70d25aeb29cba051d73d79aacd6a2`
`git status` empty before and after the run. Evidence runner was not modified to change a result.

GPU: H100-94GB (95830 MiB). The 32768-row measurement ran; it was not skipped for memory.

## pytest

Default build (`TORCH_CUDA_ARCH_LIST=9.0`, `setup.py build_ext --inplace`, two GPUs):

| passed | failed | skipped |
| --- | ---: | ---: |
| 371 | 0 | 34 |

Skip reasons: 15 SM90 kernels not in that build, 18 Ascend kernels, 1 CUDA dispatch case.

SM90 rebuild (`KERNEL_ALIGN_FORCE_SM90=1` after removing `build/`, because a stale build directory does not pick up the flag). `batch_invariant_logp_sm90` is exported. The previously skipped SM90 tests:

| passed | failed | deselected |
| --- | ---: | ---: |
| 18 | 0 | 116 |

## benchmark, 32768 rows x 151936, BF16

From `report.json` on this machine. Not copied from the B200 report.

| implementation | forward + backward | backward only | peak |
| --- | ---: | ---: | ---: |
| torch `log_softmax` + gather | 111.7 ms | 59.3 ms | 55.64 GiB |
| previous full FP32 VJP | 103.2 ms | 87.1 ms | 46.37 GiB |
| chunked fallback | 126.0 ms | 110.3 ms | 10.27 GiB |
| fused kernel | 38.1 ms | 21.8 ms | 9.27 GiB |
| Liger CE | 25.6 ms | 13.4 ms | 9.27 GiB |
| FLA CE | 15.2 ms | 10.2 ms | 9.27 GiB |
| flash-attn CE | 14.2 ms | 10.1 ms | 9.27 GiB |

Fused-kernel gradient vs the shared FP64 reference: max abs 3.87e-4, correctly rounded 99.9991%. Batch-size sweep: 0 / 0 of 2232 for logp and `dlogits` on every candidate, including FLA. The stored B200 report is a different head and a different GPU; its FLA sweep is not reused here.

## commands

```
TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -p no:cacheprovider -q -rs <logp suites listed in pytest_fused_logp.log>
CUDA_VISIBLE_DEVICES=0 python benchmarks/fused_logp_backward_evidence.py --flash-attn-src <flash-attention 94e22c9> --out report.json
KERNEL_ALIGN_FORCE_SM90=1 python setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -p no:cacheprovider -q -rs <sm90 selectors in pytest_sm90_only.log>
```
