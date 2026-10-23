# Draft: `stochastic_transition_contract` (maintainer review)

**Status:** draft on branch `plan/h3-verl-omni-e2e` only. Do not treat as merged RFC row until approved.
**Do not post to GitHub until the user approves.**

## Problem

Public H3 Diffusers sampling uses **deterministic ODE** (`eta=0`). VeRL-Omni FlowGRPO uses a **stochastic**
discrete SDE step (`sde_type=cps`, configurable `noise_level`). RFC #420 states transition log-prob fields
must stay absent until an explicit probability model is approved.

## Scope

- **In scope:** one-step reverse transition used in training and rollout for FlowGRPO-style H3 pipelines.
- **Modality:** separate video and audio latent schedulers; combined scalar loss uses fixed weights (0.5/0.5 in current recipe).
- **Out of scope:** decoded media likelihood; autoregressive logprob; ODE-only public sampler.

## Pinning (VeRL-Omni reference)

- Code entry: `FlowMatchSDEDiscreteScheduler.sample_previous_step` with `return_logprobs=True`.
- Actor path: `sample_h3_transition` in `minimax_h3_flow_grpo` training adapter.
- Rollout path: vLLM-Omni rollout adapter feeding the same scheduler semantics.
- Recipe pins: `sde_type=cps`, `noise_level=0.8`, window size 2 (as in tiny smoke).

## Contract requirements (strict / RL-Kernel)

1. **Fixed RNG stream:** noise for step `t` derived from a declared seed + `(step_index, modality, batch_row)`; no topology-dependent reordering.
2. **FP32 accumulation** for mean/variance of the transition; one declared cast to sample dtype.
3. **Fail-closed:** unsupported `sde_type`, missing sigma table entry, or shape mismatch → hard error.
4. **Batch invariance:** row `i` bytes independent of batch size and of other rows' values.
5. **Log-density:** Gaussian (or declared) density of the sampled `prev_sample` given `prev_sample_mean` and `std_dev_t` returned by the scheduler math; same formula on train and rollout.
6. **Readback:** log backend id, sigma index, `std_dev_t` hash, noise draw id.

## Deliverables after approval

1. Operator doc under `docs/operators/` with formulas matching this draft.
2. CUDA/Triton or Python strict implementation registered as `stochastic_transition_contract`.
3. Tests: vs FP64 reference on short grids; batch invariance; train/rollout hook parity on frozen tensors.
4. VeRL-Omni E2E: enable logprob comparator only when this row is merged.

## Open questions for maintainers

1. Is **CPS** the canonical noise law for the RFC row, or should we support both `cps` and `sde` under one contract?
2. Should video and audio transitions be **independently sealed** artifacts (recommended) or one combined logprob?
3. Acceptable tolerance before strict merge: **bitwise zero** only, or FP64 reference threshold for logprob?

## Suggested issue comment (post-approval template)

> Proposed contract for `stochastic_transition_contract` aligned with VeRL-Omni FlowGRPO (`cps`, dual schedulers):
> [link to this file on `plan/h3-verl-omni-e2e`]. Request review before implementation PR.
