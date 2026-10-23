# RFC #420 closeout roadmap (prepare first, communicate after)

This document is the **RFC-first** track. VeRL-Omni end-to-end work stays in [`TASKS.md`](TASKS.md) but runs **after**
the RFC acceptance story is defensible. The maintainer-facing package is outlined in
[`maintainer-proposal-outline.md`](maintainer-proposal-outline.md).

**Issue:** [RL-Align/RL-Kernel#420](https://github.com/RL-Align/RL-Kernel/issues/420)  
**Public claim comment:** [comment on #420](https://github.com/RL-Align/RL-Kernel/issues/420#issuecomment-6340566107) (stack #567–#575).

## Definition of “RFC complete” (for our proposal)

Maintainers can merge our stack when **all** of the following hold:

1. **Rows we claimed** are implemented, tested, documented, and merged into `test-h3` (or explicitly scoped as optional follow-ups with maintainer OK).
2. **WS1 / WS2 exit criteria** in RFC §7 are evidenced on Hopper (H100): real block, refiner + 50L where weights allow, dual trajectory, TP/CP/SP byte equality where topology permits.
3. **§6.3 controls** H-C0–H-C7: each either **passed** with a linked artifact or **N/A** with a one-line rationale (see [`rfc-gap.md`](rfc-gap.md)).
4. **§6.4 arms:** at minimum **P/P** and **R/R** on the **Diffusers strict stack** (#571 dual trajectory); R/P and P/R documented as localization arms (same methodology as #377).
5. **Final rows** we intend to own: `final_e2e_diffusers_cuda`, `final_distributed_h3_run`, `cuda_perf_closeout` — either merged or presented as a single follow-up PR with evidence plan (not VeRL-Omni yet).
6. **Dependencies on others’ rows** are explicit: interim/replay bindings labeled until their PRs land; no silent production fallbacks in strict traces.

VeRL-Omni P/P vs R/R is **Phase 2** (community integration proof), cited in the maintainer proposal as motivation, not as an RFC merge blocker unless maintainers say otherwise.

## PR stack status (claimed rows)

Merge order (each PR targets `test-h3`, stacks on prior head):

| Step | PR | RFC rows (summary) | Merge blocker / note |
| --- | --- | --- | --- |
| 0 | [#539](https://github.com/RL-Align/RL-Kernel/pull/539) | `ws1_one_h3_block` (prior claim) | Still open; stack rebased on `test-h3` incl. #553 |
| 1 | [#567](https://github.com/RL-Align/RL-Kernel/pull/567) | fingerprint, trace, trajectory artifact | CI green; await review |
| 2 | [#568](https://github.com/RL-Align/RL-Kernel/pull/568) | audio/text/QKV projections | CI green |
| 3 | [#569](https://github.com/RL-Align/RL-Kernel/pull/569) | WS2 TP/SP/CP + hierarchical collective | 8-GPU single-node tests skipped; 2×4 H100 evidence in PR |
| 4 | [#570](https://github.com/RL-Align/RL-Kernel/pull/570) | `token_refiner_block` | CI green |
| 5 | [#571](https://github.com/RL-Align/RL-Kernel/pull/571) | full transformer, dual trajectory | Needs full weights for strongest claim |
| 6 | [#574](https://github.com/RL-Align/RL-Kernel/pull/574) | runtime, ablation, Diffusers adapter, LoRA gate | `stochastic_transition_contract` doc pending approval |
| 7 | [#575](https://github.com/RL-Align/RL-Kernel/pull/575) | checkpoint mapper, vLLM-Omni adapter | Binds #553 attention in strict path |
| — | [#576](https://github.com/RL-Align/RL-Kernel/pull/576) | *optional perf; does not claim `h3_full_attention`* | **No #553 issue comment**; reviewers on #576 only; separate from stack merge order |

**Action:** refresh each PR body with current short SHAs and `pr_assets` commit after every evidence rerun.

## Rows we did **not** claim (external dependencies)

| Row | Owner | Impact on our stack |
| --- | --- | --- |
| `h3_full_attention` | RichApple123 (#553 merged) | Strict path uses #553; optional flash #576 |
| `h3_attention_o_gemm`, `h3_ffn_gate_up_gemm`, `h3_ffn_down_gemm` | zzf2024 | Interim `TritonDetGemmOp` in block until merged; **perf closeout blocked** |
| `h3_qk_rmsnorm_d128`, `h3_mm_rope_3axis_partial`, `h3_swiglu` | Jungle430 | Registry/replay until merged |
| `video_*`, `packed_index_*`, etc. | others | Out of our stack scope |
| `h3_sigma_schedule`, `h3_ode_step` | fxyan, Jerry536-coco | Dual trajectory uses reference scheduler until merged |
| `stochastic_transition_contract` | **us (draft)** | Required before RL logprob fields; see B2 draft |

## Work packages (RFC track R1–R8)

Update status in [`RFC-TASKS.md`](RFC-TASKS.md).

| ID | Work | Outcome for maintainer |
| --- | --- | --- |
| R1 | PR stack CI + DCO + review hygiene | All #567–#575 green; linear signed history |
| R2 | Evidence refresh on `pr_assets` | One commit SHA cited in all PRs; charts for accuracy + perf |
| R3 | Full HF weights + #571 rerun | 50L forward/backward + dual trajectory on real checkpoint (**weights on disk: 14/14 shards**) |
| R4 | H-C0–H-C7 evidence matrix | Single table with pass / gap / N/A |
| R5 | §6.4 arms on Diffusers stack | P/P vs R/R + R/P + P/R scripts and JSON (not VeRL-Omni) |
| R6 | `h3_ablation_overlay` H1–H17 table | At least one mismatch signature demo per category (or deferred list) |
| R7 | Final three rows PR | `final_e2e_diffusers_cuda`, `final_distributed_h3_run`, `cuda_perf_closeout` |
| R8 | **`maintainer-proposal-outline.md` → final PDF/Markdown** | Send after R1–R7 credible; includes VeRL-Omni Phase 2 ask |

## Performance honesty (RFC `cuda_perf_closeout`)

Current H100 picture (strict vs provider, stack with #553):

- Attention: large gap at long seq; optional flash (#576) reduces but does not beat SDPA.
- Block: dominated by interim O / FFN GEMMs until zzf2024 rows or our strict GEMM PRs land.

**Closeout strategy for maintainers:**

1. Document **correctness first** (byte drift gates met on R/R Diffusers path).
2. Report perf with **stage breakdown** (attention vs GEMM vs launch-bound nodes).
3. Ship perf improvements as **follow-up PRs** (tile pick, flash optional backend, CUDA graphs) without blocking row merges if maintainers accept “correctness closeout now, perf iterative”.

## Suggested maintainer conversation sequence

1. **Now (prepare):** keep RFC comment accurate; no new claims without user OK.
2. **When stack is review-ready:** one consolidated comment listing PR order + evidence SHA + known deps on zzf2024/Jungle430.
3. **After first merges:** post H-C matrix + dual trajectory summary.
4. **After R7:** publish closeout PR and attach [`maintainer-proposal-outline.md`](maintainer-proposal-outline.md) (Phase 2 VeRL-Omni ask).

Do **not** post Phase 2 VeRL-Omni proposal until Diffusers RFC gates in §7 are documented — otherwise reviewers conflate kernel RFC with orchestrator integration.
