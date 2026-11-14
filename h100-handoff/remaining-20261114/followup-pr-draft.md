TITLE: [WS1][MiniMax-H3] Review follow-up for #568 / #570 / #571 / #574: fail-closed projection dispatch, fused QKV backward, observed-only trace, audit fallback scope, differentiable CPS log-prob (RFC #420)

BRANCH: fix/h3-merged-rows-review  ->  base test-h3

---

<!-- Type: operator -->

> **RFC #420 H3 review follow-up** (base `test-h3`). #568, #570, #571 and #574 were merged into `test-h3` (at `cfec0d98`) while their last OpenCodeReview / CodeRabbit round was still open. This PR carries the fixes for that round, on top of the merged code. #569 and #575 stay open and do not depend on this PR; each carries its own fixes.

## What

Six commits. Each fixes review findings on code that is already in `test-h3`. Forward bytes are unchanged everywhere. What changes is listed under Results.

| Commit | Fix | Findings |
|---|---|---|
| `ff395298` | `OperatorCall.fallback` defaults to `None` (not observed). Only an explicit report claims "no fallback", and `None` survives `dumps` / `loads` as distinct from `False`. | [#570 OCR](https://github.com/RL-Align/RL-Kernel/pull/570#discussion_r4479133790) |
| `25e2b8d2` | Projection rows (#568): see the list below. | [#568 CR ×4](https://github.com/RL-Align/RL-Kernel/pull/568#discussion_r4479192983), [#569 CR on #568's files](https://github.com/RL-Align/RL-Kernel/pull/569#discussion_r4479202638) ([2](https://github.com/RL-Align/RL-Kernel/pull/569#discussion_r4479202667), [3](https://github.com/RL-Align/RL-Kernel/pull/569#discussion_r4479202717), [4](https://github.com/RL-Align/RL-Kernel/pull/569#discussion_r4479202743)) |
| `1a118e81` | The token refiner's `q_proj` / `k_proj` / `v_proj` share one fused call, as in the block graph. | same issue as the #568 QKV finding |
| `7033fbfc` | `H3DiffusersAdapter.active()` counts only a failure to *bind* the graph as an audit fallback. Audit then runs the provider. An error raised by the caller's own forward or backward propagates unchanged. | [#575 OCR on #574's `runtime.py`](https://github.com/RL-Align/RL-Kernel/pull/575#discussion_r4479151148) |
| `e3b386ec` | `fixed_order_mean` is an `autograd.Function`, so the CPS `log_prob` carries the policy gradient to the velocity. Before, it had no `grad_fn`. | [#575 OCR](https://github.com/RL-Align/RL-Kernel/pull/575#discussion_r4479151150), [#575 CR](https://github.com/RL-Align/RL-Kernel/pull/575#discussion_r4479633235) |
| `0264a4f8` | `lora_gradient_gate`'s `base_frozen` is a real check. `transition()` refuses `sample_ids` that do not match the batch. The doc gives the `VERL_OMNI_PATH` condition for the scheduler comparison. | [#575 CR](https://github.com/RL-Align/RL-Kernel/pull/575#discussion_r4479633224) ([2](https://github.com/RL-Align/RL-Kernel/pull/575#discussion_r4479633210), [3](https://github.com/RL-Align/RL-Kernel/pull/575#discussion_r4479633096)) |

What `25e2b8d2` changes in the projection rows (#568):

- **Fail-closed dispatch.** The CUDA priorities of `audio_input_projection_fp32`, `audio_output_projection_fp32`, `text_context_projection` and `h3_qkv_gemm` list only the fixed-order Triton op. A CUDA request without Triton raises instead of returning the `F.linear` reference.
- **Fused QKV backward in the block graph.** `q_proj` / `k_proj` / `v_proj` share one call of the row's combined operator (`h3_block.fused_qkv`). The block's input gradient is therefore the fused `[Q; K; V]` backward that the row documents (FP32 over all three, one rounding), instead of three BF16-rounded gradients added by autograd.
- **Reference rows take the CUDA signature.** `logical_keys` is accepted on every method and validated by the same `canonical_row_order`, which now lives in the reference module. Padding rows give no parameter gradient. `NativeH3QKVGemmOp.project` is added.
- **All-padding call.** A call whose rows are all padding returns zero parameter gradients instead of raising in the backward GEMM.
- **Docs.**
  - The keyed-gradient guarantee is per call: a batch split over several backward calls is not covered.
  - The H100 performance table and paragraph now match the linked `bench.json`.
  - The bench docstring says forward only.

## Prior art & reuse decision

There is no new operator and no new kernel. The fixes reuse what the merged rows already have:

| Need | Reused |
|---|---|
| fused QKV input gradient | `H3QKVGemmCudaOp.forward` / `_QKV` backward (`h3-projection-gemm-v1`, #568) |
| differentiable fixed-order mean | the same fixed-order Triton GEMM for the forward. The backward is element-wise `grad / K`, the same values `torch.mean`'s backward gives. |
| fail-closed dispatch | the existing registry semantics (`get_op` raises when no candidate constructs), as the merged video projection rows (#562, #646) already use |
| key validation in the reference | `canonical_row_order` from the CUDA rows, moved to the reference module and re-exported |

## Results

**Unchanged:**
- forward bytes of every node and of the CPS transition;
- the operator trace format, apart from the `fallback` default;
- the merged evidence of #568 (forward time and accuracy per row).

**Changed:**

| Where | Before | Now |
|---|---|---|
| block and refiner input gradient at `norm1` | three projections, each `dX` rounded to BF16, added by autograd | one fused `[Q; K; V]` backward, FP32 accumulation, one rounding |
| CPS `log_prob` | no `grad_fn`: a policy loss got no gradient through it (checked on GPU: `requires_grad=False`) | differentiable. The velocity gradient equals `torch.mean` autograd of the same formula (rtol 1e-6), and each sample's gradient is bitwise the same alone as in a batch. |
| `h3_qkv_gemm` and the three other rows on CUDA without Triton | returned the `F.linear` (cuBLAS) reference | `RuntimeError: No functional backend` |
| audit mode, error inside the bound step | recorded as `fell_back` and reported `P/P` | propagates. `fell_back` is set only when binding fails, and the run then continues on the provider. |
| observer traces | `fallback=False` by default | `fallback not observed` until an operator reports it |

The full-model gradient numbers in #571 and #574 (measured on H100 and B200) were taken before the fused QKV backward, so their input-gradient columns would move. They are **not re-measured** here (see Status).

## Status

- **Head `0264a4f8`** on `fix/h3-merged-rows-review`, base `test-h3` `cfec0d98`. Six commits, no merge commits, every commit signed off.
- **Not re-measured:** the 33B strict-vs-provider gradient columns of #571 / #574 and the dual trajectory with the fused QKV backward. They need real weights; this host had none.
- **Review threads:** each finding linked under What gets a reply that points here.

**History**

- **2026-11-05: opened.** #568, #570, #571 and #574 were merged at their morning heads while this review round was open. The fixes were first made on those branches and are carried here unchanged, one commit per review topic.

## Tests

`pytest tests/models/minimax_h3`, default build (`KERNEL_ALIGN_USE_FAST_MATH=0`), on 2 × NVIDIA B200 (sm100). Driver 580.126, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1. `_C` and `_H3_C` were built on that host from the same native sources (`RL_KERNEL_H3_STRICT=1`; no native file changes in this PR). **No H3 weights on the host**, so every real-weight test skips. `test-h3` at `cfec0d98` is the baseline on the same host and build.

| Commit | Command | passed / skipped / failed |
|---|---|---|
| `0264a4f8` (this PR) | `pytest tests/models/minimax_h3` | {FOLLOW_ALL} |
| `0264a4f8` | the five changed test files (below) | {FOLLOW_PER} |
| `cfec0d98` (`test-h3`, baseline) | `pytest tests/models/minimax_h3` | {BASE_ALL} |

New or changed tests:
- `test_h3_operator_trace.py`: unreported fallback is `None`; `None` round trip.
- `test_h3_projection_rows.py`: CUDA dispatch fails closed without Triton; reference signature and key contract; all-padding zero gradients.
- `test_h3_one_block.py` and `test_h3_token_refiner.py`: the three QKV nodes make one combined call; a copied context recomputes.
- `test_h3_runtime_integration.py`:
  - an error inside the bound step is not a fallback (strict and audit);
  - audit runs the provider when binding fails, and strict raises;
  - the log-prob carries the policy gradient;
  - the LoRA gate detects a gradient reaching the base;
  - mismatched `sample_ids` are refused.

Each new test was checked to fail on the code before its fix. The `KERNEL_ALIGN_USE_FAST_MATH=1` build was not run.

## Scope / Known limitations

- The fused QKV backward was checked structurally (one combined call per block) and through the unit tests. It was not checked against real block weights and the one-block FP64 bounds, because no weights were available. A real-weight run of `test_h3_one_block.py` / `test_h3_token_refiner.py` is the open check.
- The reference rows keep the provider's token-axis reduction order. Only key validation and the padding semantics match the CUDA rows.
- A canonical accumulator across several backward calls is not part of the projection rows.
- Not here: the fixes for #569's and #575's own code. They are on those PRs.
