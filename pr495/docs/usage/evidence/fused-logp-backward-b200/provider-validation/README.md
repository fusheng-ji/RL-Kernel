# Actual Vime and vLLM WS2 provider validation

The provider experiment found two distinct limitations. Existing Vime training and
vLLM WS2 rollout agree exactly for the same batch, including training TP=2 versus
rollout scoring TP=1. Their outputs change under some chunking and active-row
compaction cases. Using flash-attn CE on **both** sides passes the tested TP=1
contract, but training TP=2 versus full-vocabulary rollout TP=1 produces differences
in both logp and LSE. The complete existing-provider baseline verdict is therefore
**false**, and the tested CE composition also fails the cross-TP replacement gate.

[index.json](index.json) contains the readable summaries, provenance, checksums, and
input-identity checks. [tp1.json.gz](tp1.json.gz) and [tp2.json.gz](tp2.json.gz) contain
the complete original reports, compressed losslessly without dropping records.
The source tree was clean at `3c49083dcbb18ce596a733de372f35fceb97dc9b` for both runs.

## What actually ran

The runner calls Vime's `provider(request)` and an unmodified
`VllmLogpOperator(Sampler.forward)`, using the installed vLLM `Sampler` in
`raw_logprobs` mode with greedy sampling. Token IDs come from that real sampler.
Both integration adapters actually dispatch `pytorch-vocab-parallel-logp-ws2`, with
`fallback=false` and 64 fixed global vocabulary tiles. This is the backend selected
by these adapters; no execution of a CUDA WS2 replacement backend is claimed here.

The adapters return selected logp but hide LSE from their public results. A temporary
`sys.setprofile` return-frame observer reads their actual `_lse` tensors. It restores
the prior profiler and does not replace a provider, registry entry, numerical kernel,
or collective. The reports include raw output bytes, numerical readbacks, installed
vLLM source hashes, and hashes of each full scoring input and training shard.

For the CE candidate, the same pinned flash-attn `cross_entropy_loss` implementation
scores the real sampler's token IDs on both sides. Training uses local vocabulary
shards and its real distributed process group; rollout uses the full replicated
vocabulary with no process group. This is an experimental scorer composition after
actual sampling, rather than an installed flash-attn production rollout provider.
Its actual internal LSE is observed at `CrossEntropyLoss.forward` return.

The environment was PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1, Python 3.12.14,
and vLLM 0.30.0. The CE source is pinned to flash-attention commit
`94e22c906678e5483fa0e9e24d8e787bc2c0ed4c`, file SHA256
`cb9ea79ae3ada5237133bb0365c512817288b0811ac9fbafad4f1b015a20fe7c`.
`VLLM_USE_FLASHINFER_SAMPLER=0` selects the native sampler configuration.

TP=1 uses one B200. TP=2 uses two distinct physical B200s, one process per GPU, with
NCCL transport. The vLLM wrapper scores a full vocabulary at TP=1 even in the TP=2
training experiment; that asymmetry is recorded explicitly. Both TP=2 ranks produce
the same reported counts below. No TP=4 result or four-GPU topology is claimed.

## Cases and denominator

Each topology has 48 configurations and 1,200 full-batch token rows:

- Seeds 3, 4, and 5; batch sizes 17 and 33.
- BF16 and FP32 execution logits; temperatures 1.0 and 0.7.
- Padded vocabulary 151936, with either real vocabulary 151936 or 151929.
- The latter cases also apply a shared support mask, setting columns whose index
  modulo 7 equals 1 to `-inf`, and exercise active-row compaction.

Temperature is applied once by the shared caller as an FP32 divide followed by an
execution-dtype cast. The sampler receives temperature 1.0 and scores those frozen
inputs in raw mode. Padding and support masks are likewise shared caller inputs;
this does not claim native top-p replay semantics. Vime receives explicit real/padded
vocabulary metadata, while vLLM's wrapper declares the full width and receives the
same `-inf` padding.

Every configuration checks the full batch, reverse permutation, and chunks of one
and seven rows. Masked cases also remove inactive rows and call the actual providers
on the remaining rows: 456 active-row comparisons per output kind. A separate real
WS2 registry call exercises `MaskSpec.active_mask` and ignored targets because the
integration adapters themselves declare all rows active.

Matched configurations have identical full-input hashes and actual sampled-token
readbacks across TP=1 and both TP=2 ranks. Their full-batch WS2 logp and LSE readbacks
are also byte-identical across those runs.

## Existing WS2 provider baseline

Counts below apply separately to TP=1 and to **each** TP=2 rank. The chunking and
compaction rows apply to each of the two existing providers. Outputs are FP32 and
comparisons use raw bytes, including signed zero.

| Check | Differing logp rows | Differing LSE rows | Maximum absolute difference |
| --- | --- | --- | --- |
| Vime training versus vLLM rollout, full batch | 0 / 1200 | 0 / 1200 | 0 |
| Vime training versus its `no_grad` call | 0 / 1200 | 0 / 1200 | 0 |
| Reverse permutation versus original full batch | 0 / 1200 | 0 / 1200 | 0 |
| One-row chunks versus original full batch | 22 / 1200 | 22 / 1200 | 1.9073486328125e-6 |
| Seven-row chunks versus original full batch | 27 / 1200 | 27 / 1200 | 1.9073486328125e-6 |
| Compacted active rows versus original active rows | 6 / 456 | 6 / 456 | 9.5367431640625e-7 |

Within each identical layout, Vime and vLLM still agree with one another. Changing
the layout changes their common result in the listed cases. All direct active-mask
TRAIN/INFER comparisons agree, and inactive selected logp is exactly zero.

The combined baseline summary counts **220 differing scalar checks out of 34,224
per rank**, across both output kinds, providers, and repeated layout controls.
`existing_ws2_provider_baseline.passes` is false in both topology reports. All
observed outputs are finite. The matching full-batch controls do not establish
batch/chunk invariance for this backend in this environment.

An independent reproduction isolated the existing PyTorch tile reduction. For seed
3, 17 BF16 rows, temperature 1.0, real vocabulary 151929, and the support mask above,
the full batch and singleton calls have identical sampled IDs and tile maxima, but
349 / 1088 tile sum-exp entries differ. For row 0, tile 2, the exponentiated inputs
are byte-identical while `torch.sum` returns `4.893764495849609` in the batch and
`4.893764972686768` alone. Row 0 LSE becomes `13.793819427490234` versus
`13.79382038116455`. A PyTorch-only reproduction produces the same discrepancy
without either provider or the observer. PyTorch's per-row reduction tree changes
with the output-row count; fixing the vocabulary tiles alone does not fix that tree.
No production change to this existing WS2 reduction is included in this experiment.

## Coordinated flash-attn CE candidate

A difference between CE and the previous WS2 implementation is a change-of-arithmetic
diagnostic. A coordinated replacement can change both providers together. The
replacement gate therefore compares CE training with CE rollout on the same frozen
inputs and actual sampled IDs, while separately checking each side's own invariance.

| Training TP → rollout scoring TP | Differing logp rows, full batch | Differing LSE rows, full batch | Maximum absolute difference |
| --- | --- | --- | --- |
| 1 → 1 | 0 / 1200 | 0 / 1200 | 0 |
| 2 → 1, each training rank | 282 / 1200 | 282 / 1200 | 1.9073486328125e-6 |

The TP=2 mismatch persists after reversal and either chunk size: each layout also
has 282 / 1200 differing rows for each output kind. The combined coordinated summary
therefore has 2,256 differing scalar checks out of 9,600 per TP=2 rank, including the
full batch and those three layouts. The two-rank aggregate repeats these observations;
it represents the same 48 configurations observed on both ranks.

The following CE controls all have zero byte differences at both tested TP degrees:

| CE control | Logp differences | LSE differences |
| --- | --- | --- |
| Training versus same-provider `no_grad` | 0 / 1200 | 0 / 1200 |
| Replicated full-vocabulary TP=1 training versus rollout | 0 / 1200 | 0 / 1200 |
| Each side's reverse permutation | 0 / 1200 | 0 / 1200 |
| Each side's one-row chunks | 0 / 1200 | 0 / 1200 |
| Each side's seven-row chunks | 0 / 1200 | 0 / 1200 |
| Each side's active-row compaction | 0 / 456 | 0 / 456 |

For completeness, CE versus WS2 differs in 42 / 1200 rows per output kind at TP=1
and 272 / 1200 at TP=2. These diagnostic counts are excluded from the replacement
verdict. The measured reason the tested coordinated CE composition fails is its
**TP=2 training versus TP=1 rollout** difference despite passing the local controls.
The sampled TP=1 CE composition passes its measured consistency gates.

This evidence also records an existing WS2 layout failure. It does not establish
that either implementation satisfies every strict provider contract, and it does
not validate a production replacement or a tolerance-based policy change.

## Scope and reproduction

The new runner's regression suite passed all 12 tests, including actual CUDA/vLLM
provider calls, raw-byte comparison accounting, and replacement-verdict checks.
Black 24.4.2, isort 5.13.2, and flake8 7.0.0 checks also pass.

This experiment exercises actual integration providers and actual sampler execution
on frozen logits. It does not construct a model, a vLLM serving engine, or an
end-to-end training/rollout job. It measures forward logp/LSE consistency; the separate
[shared-gradient accuracy report](../review-report.json) covers the earlier backward
accuracy rerun. No performance measurements were made.

Every configuration also sends an actual top-p keep-mask request to Vime's WS2
provider and observes `SelectedLogprobProviderUnavailable`. That unsupported request
is recorded as a rejection; top-p replay correctness is not validated.

From the repository with the recorded environment and pinned flash-attention source:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  python benchmarks/fused_logp_provider_evidence.py \
  --out <TMP>/x --seeds 3 4 5 \
  --flash-attn-src <TMP>/x/flash-attention
CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  torchrun --standalone --nproc-per-node=2 \
  benchmarks/fused_logp_provider_evidence.py \
  --out <TMP>/x --seeds 3 4 5 \
  --flash-attn-src <TMP>/x/flash-attention
```

The committed gzip files preserve the original JSON bytes. Read either with
`gzip.open`, or decompress it before inspection:

```python
import gzip
import json

with gzip.open("tp2.json.gz", "rt") as handle:
    report = json.load(handle)
print(report["summary"])
```

The smaller command below reproduces the existing WS2 chunking case, including the
masked configuration and default one-row/seven-row layout checks:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  python benchmarks/fused_logp_provider_evidence.py \
  --out <TMP>/x --seed 3 --rows 17 \
  --dtypes bfloat16 --temperatures 1.0 --vocab 151936 --padding 7
```
