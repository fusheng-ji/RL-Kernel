# h3_qk_rmsnorm_d128, B200 evidence

Measured at `944a585d` (clean tree) on one NVIDIA B200, torch 2.13.0+cu130.

| Path | What |
|---|---|
| `h3-prior-art-b200/qk_norm.json`, `.png` | batch invariance, error against FP64, latency: this row, the provider, the generic RMSNorm operators, TE, Liger, SGLang |
| `h3-bitwise-reuse-b200/qk_norm.json`, `.png` | byte comparison of every candidate with the provider and with this row |
| `tests-b200/` | the test log, the base's log for the three failing tests' files, and the two runner logs, with paths replaced |

## H100 (2026-11-26)

Measured at `c56b81c4` (clean tree) on one NVIDIA H100, torch 2.13.0+cu130; paths replaced by `<E>`, `<W>`, `<OUT>`.

| Path | What |
|---|---|
| `h3-prior-art-h100/qk_norm.json` | batch invariance, error and latency (TE, Liger, SGLang not installed there) |
| `h3-bitwise-reuse-h100/qk_norm.json` | byte comparison with the provider and with this row |
| `tests-h100/` | test log (532 passed, 1 failed, 56 skipped), runner logs, `_C` location, `nvidia-smi` |

## H100 follow-up (2026-11-27)

At `c56b81c4` on one NVIDIA H100, the same build; paths replaced by `<HOME>`, `<WORK>`, `<E>`.

| Path | What |
|---|---|
| `tests-h100/qk-weights.txt` | `test_h3_qk_rmsnorm.py` with block 0's real weights set: the two checkpoint tests pass |
| `tests-h100/ws1-evidence-head.txt`, `ws1-evidence-base.txt` | `test_ws1_candidate_evidence.py` at this commit and at the base `13d0f906`: the same `det_gemm` failure on both |

## B200 after the merge-up (2026-11-27)

At `5abdf974` (merge of `test-h3` `60c6aaf9`) and at the base `60c6aaf9`, one NVIDIA B200, clean trees, `_C` built per tree; paths replaced by `<WORK>`, `<USER>`, `<TMP>`.

| Path | What |
|---|---|
| `tests-b200-mergeup/tests-head.txt` | 567 passed, 3 failed, 33 skipped |
| `tests-b200-mergeup/tests-base.txt` | the same set without `test_h3_qk_rmsnorm.py`: 526 passed, the same 3 failed, the same 33 skipped |
