# RFC #420 gap analysis

Reference: [issue #420](https://github.com/RL-Align/RL-Kernel/issues/420).

**RFC-first:** gates below are filled for **Diffusers strict stack** (#571) before VeRL-Omni Phase 2.
Roadmap: [`rfc-closeout-roadmap.md`](rfc-closeout-roadmap.md). Tasks: [`RFC-TASKS.md`](RFC-TASKS.md).

Snapshot: **2026-10-23** (issue table + claim comments; CUDA scope only).

## CUDA row ownership (for maintainer proposal)

Issue #420’s table still shows many rows as `OPEN` with an empty Survey column even after claim comments.
Use the three tables below when drafting the closeout message.

### A. Unclaimed on CUDA path (no assignee in comments; we may claim)

| Row | Notes |
| --- | --- |
| `stochastic_transition_contract` | Draft in [`stochastic-transition-contract-draft.md`](stochastic-transition-contract-draft.md); **claim after user approval** |
| `final_e2e_diffusers_cuda` | Pure CUDA final gate; planned as R7 |
| `final_distributed_h3_run` | Multi-step TP4/CP2 sealed run; R7 |
| `cuda_perf_closeout` | Pure CUDA perf row; R7 / B3 |
| `sglang_runtime_adapter_h3` | **Out of scope** unless maintainers request it |
| `ws2_pipeline_parallel` | Optional follow-up; **out of scope** for first closeout |
| Phase B rows (`qwen3vl_*`, `*_vae_*`, `fl2va_*`, `ref2va_*`, `phase_b_full_chain`) | **N/A** for our VeRL-Omni CUDA path |

### B. Contested or table out of sync (needs maintainer assign)

| Row | Parties | Our PR / comment |
| --- | --- | --- |
| `audio_input_projection_fp32`, `audio_output_projection_fp32` | **walker-ai** (2026-10-22) vs **fusheng-ji** [#568](https://github.com/RL-Align/RL-Kernel/pull/568) in [stack comment](https://github.com/RL-Align/RL-Kernel/issues/420#issuecomment-6340566107) | Resolve before merge; video projections stay with walker-ai |

### C. Claimed by others — CUDA dependency (not unclaimed; blocks strict perf / parity)

| Row | Claimant | Impact if not merged |
| --- | --- | --- |
| `h3_attention_o_gemm`, `h3_ffn_gate_up_gemm`, `h3_ffn_down_gemm` | zzf2024 | Interim GEMM in block; **`cuda_perf_closeout`** |
| `h3_qk_rmsnorm_d128`, `h3_mm_rope_3axis_partial`, `h3_swiglu` | Jungle430 | Strict replay / fusion parity |
| `packed_index_copy`, `conditioning_noise_mix` | ThreeMonth03 | Packed IO / conditioning |
| `packed_index_select` | Jungle430 | Row select + grad scatter |
| `h3_sigma_schedule` | fxyan | Dual trajectory / scheduler grid |
| `h3_ode_step` | Jerry536-coco | Deterministic Euler reference |
| `video_input_projection_fp32`, `video_output_projection_fp32` | walker-ai | Video FP32 projections |
| `h3_packed_layout_schema` | BPbruce | Layout schema |
| `h3_full_attention` | RichApple123 | **Merged** (#553) |

### D. Claimed by us — stack comment + PRs (merge pending)

See [comment #6340566107](https://github.com/RL-Align/RL-Kernel/issues/420#issuecomment-6340566107): #567–#575 (+ prior #483–#494, #539).
Optional perf: [#576](https://github.com/RL-Align/RL-Kernel/pull/576) (does **not** claim `h3_full_attention`).

Early WS1/WS2 rows with `@fusheng-ji` in the issue table (`timestep_*`, `adaln_*`, `h3_rmsnorm`, `final_adaln_out`, `ws1_one_h3_block`, `tp_adaln_3mod`, `sp_norm_adaln`) map to open PRs **#483–#494**, **#539**.

## §6.4 experiment arms

| Arm | Train | Rollout | VeRL-Omni mapping | Status |
| --- | --- | --- | --- | --- |
| P/P | Production | Production | Fork `main`, native Diffusers + vLLM-Omni | Tiny smoke + TP=2 LoRA sync PASS; full weights on disk (R3 rerun) |
| R/P | RL-Kernel strict | Production | Integration branch, train hook only | Not run |
| P/R | Production | RL-Kernel strict | Integration branch, rollout hook only | Not run |
| R/R | RL-Kernel strict | RL-Kernel strict | Integration branch, both sides | Dual-trajectory on Diffusers stack (#571); not yet VeRL-Omni (B6/B7) |

## §6.3 invariant controls (H-C0–H-C7)

| Control | Required result | Evidence today | Gap |
| --- | --- | --- | --- |
| H-C0 | Repeat identical one-step call → zero drift | Flash / #553 self-repeat bitwise (operator) | Model-level repeat not in VeRL-Omni |
| H-C1 | Request alone vs batch row → zero drift | Flash batch invariance tests | VeRL-Omni micro-batch=1; extend in B6 |
| H-C2 | Same noise + full dual trajectory → zero every step | `ws1_dual_trajectory`, 3 seeds JSON in handoff | Not on VeRL-Omni schedulers |
| H-C3 | Permutation vs canonical packed order | Documented probe in `h3-one-block` (not bitwise promise) | Re-run under strict VeRL-Omni layout |
| H-C4 | TP1 vs TP2/4/8 | WS2 JSON (tp2/4/8) on stack | VeRL-Omni uses TP2 rollout + FSDP; map in B4 |
| H-C5 | CP1 vs CP | WS2 TP4×CP2 / SP evidence | Align Ulysses SP with `cp_full_attention` row |
| H-C6 | Grad checkpoint off vs on | Not summarized | Add before B7 |
| H-C7 | FL2VA/Ref2VA preprocessing repeat | Out of scope (Phase B rows declined) | N/A for VeRL-Omni path |

## Claimed / open RFC rows (fusheng-ji stack)

| Row | PR / note | Merged to `test-h3` |
| --- | --- | --- |
| h3_arch_fingerprint, h3_operator_trace, h3_trajectory_artifact | #567 | Open |
| audio/text/QKV projection rows | #568 | Open |
| ws2 parallel | #569 | Open (8-GPU help wanted) |
| token_refiner_block | #570 | Open |
| ws1_full_h3_transformer, ws1_dual_trajectory | #571 | Open |
| h3_runtime_modes, ablation, adapters, LoRA gate | #574 | Open |
| checkpoint mapper, vllm adapter | #575 | Open |
| h3-flash-attention-v1 (optional, not row claim) | #576 | Open |
| **stochastic_transition_contract** | B2 draft only | Not claimed |
| **final_e2e_diffusers_cuda** | B7 deliverable | Not claimed |
| **final_distributed_h3_run** | B7 deliverable | Not claimed |
| **cuda_perf_closeout** | B3 + B7 | Not claimed |

## Final experiment gate (RFC §7)

| Gate item | Status |
| --- | --- |
| Matched workload, seeds, sigma hashes, hardware | Designed in `experiment-design.md`; not executed |
| No missing/non-finite metrics | #377 rules adopted |
| Per-step artifacts + readback | Planned; VeRL-Omni metrics partial (logprob only today) |
| Performance split: model / scheduler / comm / encode / decode | Not implemented for H3 |
| Latent trajectory equality hard gate | H-C2 path on Diffusers; extend to VeRL-Omni |

## Ablation overlay (H1–H17)

`h3_ablation_overlay` implementation exists in #574; **no published mismatch signature table** for H1–H17
on VeRL-Omni. Add during B5/B7 when R/P and P/R arms localize faults.
