# PR #495 review rerun: shared-gradient accuracy and rollout compatibility

The [provider-level follow-up](provider-validation/README.md) now exercises the actual
Vime training provider and vLLM WS2 rollout adapter with the genuine vLLM sampler,
including physical TP=1/2 and a coordinated flash-attn CE scorer comparison.
It also records an existing WS2 reference-provider chunk-invariance failure. The
generic API comparisons below remain useful compatibility diagnostics; differences
between two implementations alone do not reject a coordinated replacement.

[review-report.json](review-report.json) contains the corrected numerical rerun for
PR #495. It was generated on 2026-10-18 from clean commit `dcbf60d621330f51de2a6359deb061d3a6aa21ab`
(`git_dirty=false`) on an NVIDIA B200, with BF16 logits and vocabulary size 151936.
The JSON records the exact command, UTC start time, runner checksum, extension checksum,
package versions, and extension build configuration.

The extension was built with CUDA 13.0 for `sm_100`, `MAX_JOBS=8`, and
`KERNEL_ALIGN_USE_FAST_MATH` unset (no `--use_fast_math`). The environment was
PyTorch 2.13.0+cu130, Triton 3.7.1, Python 3.12.14, Liger 0.8.4, and FLA 0.5.2. flash-attn's
[existing Triton CE source](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/flash_attn/ops/triton/cross_entropy.py)
was loaded directly from commit `94e22c906678e5483fa0e9e24d8e787bc2c0ed4c`;
its SHA256 is `cb9ea79ae3ada5237133bb0365c512817288b0811ac9fbafad4f1b015a20fe7c`.
This does not require building or importing flash-attn's CUDA extension.

Both B200s were already serving active workloads during the rerun. Only accuracy and
invariance checks were run; this report contains no new performance measurements.

## Shared-gradient accuracy

All seven candidates use one random upstream gradient quantized to BF16 **once** and
promoted losslessly to FP32. The runner verifies that casting these values to each
forward output dtype preserves them exactly. A single FP64 VJP reference is computed
from those shared values and the same BF16 logits, before the candidate loop.

Accuracy uses 257 rows, seed 1, and 39,047,552 gradient elements. Every candidate has
the same maximum absolute gradient error, `0.0032133508838319003`, and the same
`max(abs(error)) / max(abs(reference_gradient))`, `0.0011425249434258748`.
The earlier claimed maximum-error advantage over FP32-output implementations is
withdrawn: it came from comparing different effective upstream gradients.

| Candidate | Forward output dtype | Max absolute logp error vs FP64 | Correctly rounded BF16 gradient elements |
| --- | --- | --- | --- |
| torch log_softmax + gather | float32 | 1.1502501e-06 | 99.9921608% |
| previous VJP (full FP32 softmax) | bfloat16 | 0.057275274 | 99.9994980% |
| chunked fallback | bfloat16 | 0.057275274 | 99.9994980% |
| fused kernel | bfloat16 | 0.057275274 | 99.9986324% |
| Liger cross_entropy | bfloat16 | 0.057275274 | 74.2648041% |
| FLA fused cross_entropy | float32 | 1.3862531e-06 | 99.9918638% |
| flash-attn cross_entropy (verl) | float32 | 1.1502501e-06 | 99.9894180% |

The correctly rounded fraction compares every returned gradient element with the
shared FP64 reference rounded to BF16. Forward errors include the different output
quantization boundaries; BF16 and FP32 logp errors should be interpreted with that
dtype difference in view.

## Batch invariance

`row_invariance` compares 256 rows computed individually with those same rows in a
4096-row batch for seeds 3, 4, and 5: 768 row comparisons per candidate.
`batch_size_sweep` compares eight probe rows individually with the same rows at the
front, middle, and back of batch sizes 1..9 and 2^k-1 / 2^k / 2^k+1 through 2048,
for the same seeds: 2232 comparisons per candidate. Entries count differing
logp rows / differing gradient rows, using exact equality.

| Candidate | Row check: logp / gradient | Size sweep: logp / gradient |
| --- | --- | --- |
| torch log_softmax + gather | 0 / 0 | 0 / 0 |
| previous VJP (full FP32 softmax) | 0 / 0 | 0 / 0 |
| chunked fallback | 0 / 0 | 0 / 0 |
| fused kernel | 0 / 0 | 0 / 0 |
| Liger cross_entropy | 0 / 0 | 0 / 0 |
| FLA fused cross_entropy | 10 / 4 | 92 / 0 |
| flash-attn cross_entropy (verl) | 0 / 0 | 0 / 0 |

These are candidate-local invariance checks. They establish whether the candidate's
own arithmetic changes with the batch layout. Agreement with an existing rollout
provider is a separate requirement.

## Training and existing generic rollout compatibility

The new check compares training output against the existing generic CUDA forward APIs
`_C.fused_logp`, `_C.fused_logp_forward_fp32`, and `_C.fused_logp_forward_out`.
It tests two input regimes: BF16 logits and the same logits losslessly upcast to FP32,
as in `selected_logprobs_with_operator` before its FP32 forward call. Each regime
has 257 rows for seeds 3, 4, and 5, or 771 rows per candidate.

The FP32-input comparison controls for output dtype: the generic candidate itself
produces FP32 and agrees exactly with the existing FP32 rollout. The BF16-input
comparison instead checks the BF16 execution boundary. Each comparison uses raw
output bytes. The JSON also records the same comparisons against the FP32 `out`
entrypoint; its results agree with the allocating FP32 API.

| Candidate | FP32-input rows differing from FP32 rollout | Max FP32 absolute difference | BF16-input rows differing at BF16 boundary | Train / same-provider no_grad differing rows: BF16 / FP32 inputs |
| --- | --- | --- | --- | --- |
| torch log_softmax + gather | 51 | 1.9073486e-06 | 0 | 0 / 0 |
| previous VJP (full FP32 softmax) | 0 | 0 | 0 | 0 / 0 |
| chunked fallback | 0 | 0 | 0 | 0 / 0 |
| fused kernel | 0 | 0 | 0 | 0 / 0 |
| Liger cross_entropy | 69 | 1.9073486e-06 | 0 | 0 / 0 |
| FLA fused cross_entropy | 188 | 1.9073486e-06 | 0 | 0 / 0 |
| flash-attn cross_entropy (verl) | 75 | 1.9073486e-06 | 0 | 0 / 0 |

flash-attn CE remains batch-invariant in the tested layouts, and its training output
matches its own `no_grad` output. Against the existing generic FP32 rollout, however,
it differs in 75 / 771 FP32-input rows, with maximum absolute difference
`1.9073486328125e-06`. For BF16 inputs with FP32 outputs, it differs in 74 / 771 rows
with the same maximum difference. At the BF16 execution boundary it differs in
0 / 771 BF16-input rows.
The coarser BF16 result does not establish FP32 compatibility with an unchanged
generic provider. These differences are implementation-compatibility diagnostics,
not a test of using CE on both sides of a coordinated replacement. The
[provider follow-up](provider-validation/README.md) performs that separate experiment:
CE agrees at TP=1 but changes logp and LSE when training uses TP=2 and rollout
uses the full replicated vocabulary. The shared-gradient maximum errors are equal.

The native BF16-output generic candidate also differs from the FP32 reference when
its BF16 result is promoted to FP32. Those rows reflect an output-precision boundary;
they are not the FP32-input provider comparison used for the replacement decision.
This first rerun did not exercise the separate vLLM WS2 rollout provider; its actual
provider results are now available in the follow-up above. All results are sampled
checks of the listed shapes and seeds.

## Reproduce

Install the optional comparison packages into an isolated directory and provide the
pinned flash-attention source tree above. The rerun used `<TMP>/x`
and `<TMP>/x/flash-attention`:

```bash
python -m pip install --target <TMP>/x --no-deps \
  liger-kernel==0.8.4 fla-core==0.5.2
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<TMP>/x \
  python benchmarks/fused_logp_backward_evidence.py \
  --out <TMP>/x \
  --flash-attn-src <TMP>/x/flash-attention \
  --checks accuracy row-invariance batch-size-sweep train-rollout-invariance \
  --build-note 'Full default extension; CUDA 13.0; sm_100; MAX_JOBS=8; KERNEL_ALIGN_USE_FAST_MATH unset (no --use_fast_math)'
```

The loaded `_C` extension must expose `fused_logp_backward`. Omit `performance` on a
shared GPU. The runner saves each completed check so an interruption preserves its
actual completed results. The checked-in report contains all four requested checks.

## Regression validation

The targeted CUDA, device/indexing, ROCm-dispatch, and benchmark regressions passed:
78 tests. Related operator/dispatch coverage passed 184 tests with 33 skips.
The two-GPU negative control using the pre-fix binary failed in 2,742 / 13,403 gradient
elements (maximum absolute difference 2.16307); the isolated pre-fix loop
harness also triggered UBSan at `2147483392 + 256`. The updated cases pass.

An additional ROCm/Triton suite had 17 passes, one skip, and one failure in
`TestFusedKernelPath::test_tile_stats_read_input_dtype_exactly[triton]`. The same
failure was reproduced on baseline commit `c6a471ea`; it is unrelated to these
changes. No ROCm fused-backward build or runtime validation is claimed.

## Historical artifacts

[report.json](report.json) and [figure.png](figure.png) retain the original measurements
at `f6d3a24` for timing/memory provenance. Their accuracy comparison used different
effective upstream gradients for BF16-output and FP32-output candidates and is obsolete.
In particular, the old figure's lower-left accuracy panel must not be used to compare
candidate accuracy. The original timings predate the review fixes and were not rerun
under the shared-GPU conditions above. Use `review-report.json` for corrected accuracy,
batch invariance, and generic rollout comparisons.
