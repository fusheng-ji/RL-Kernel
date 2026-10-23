# Maintainer proposal outline (DRAFT — do not post until RFC closeout ready)

**Audience:** RL-Kernel maintainers (`@maxiaosong1124` and reviewers).  
**Language:** English (final version).  
**Timing:** After R1–R7 in [`RFC-TASKS.md`](RFC-TASKS.md) are documented; user approves before any GitHub post.

---

## 1. Executive summary

We completed (or are completing) the claimed MiniMax-H3 CUDA rows in [#420](https://github.com/RL-Align/RL-Kernel/issues/420) as a **stacked PR series** (#567–#575) on `test-h3`, with Hopper evidence and strict fail-closed semantics aligned with the Qwen3 VIME work ([#377](https://github.com/RL-Align/RL-Kernel/pull/377)).

We request maintainer guidance on:

1. Merging the stack in the published order, with explicit **interim bindings** where peer rows (O/FFN GEMM, SwiGLU, etc.) are still open.
2. Accepting a **correctness-first closeout** for `cuda_perf_closeout` while perf follow-ups (projection tile pick, optional flash attention backend #576) land incrementally.
3. **Phase 2:** community partnership to run the same P/P vs R/R methodology on **VeRL-Omni** (Diffusers train + vLLM-Omni rollout) — orthogonal to merging the kernel RFC, but central to our v0.1.0 product story.

---

## 2. What we delivered (RFC rows)

| PR | Rows | Evidence type |
| --- | --- | --- |
| #567 | arch fingerprint, operator trace, trajectory artifact | Manifest + sealed JSON |
| #568 | audio/text/QKV projections | FP64 + batch invariance + H100 bench |
| #569 | WS2 TP/SP/CP | Byte-equal vs WS1; 2×4 H100 multi-node |
| #570 | token refiner block | Block-level strict replay |
| #571 | full transformer, dual trajectory | 33B path + rollout vs train control |
| #574 | runtime modes, ablation, Diffusers adapter, LoRA gate | Integration + overlay |
| #575 | checkpoint mapper, vLLM-Omni adapter | Mapper tests + rollout hook surface |

Optional: [#576](https://github.com/RL-Align/RL-Kernel/pull/576) flash backend — **does not claim** `h3_full_attention`; offered for maintainer + #553 author review.

---

## 3. RFC §7 gates — our status (fill before sending)

| Gate | Status | Link |
| --- | --- | --- |
| Per-operator FP64 + invariance | Done in stack PRs | PR test logs + `pr_assets` |
| WS1: one block + refiner + 50L | Partial / blocked on weights | TBD after R3 |
| WS1: dual trajectory byte-equal | 3 seeds on stack; refresh on real weights | `pr_assets` / #571 |
| WS2: TP2/4/8, TP4×CP2(/SP) | Evidence in #569; 8-GPU single-node skipped | PR body |
| Strict trace: no silent production ops | Runtime readback in #574 | TBD audit script |
| Final e2e Diffusers CUDA | R7 PR | Not started |
| Perf closeout | Honest gap vs provider; plan in roadmap | [`rfc-closeout-roadmap.md`](rfc-closeout-roadmap.md) |

---

## 4. Dependencies on other contributors (no overlap)

We rely on merged or forthcoming rows from:

- **RichApple123** — `h3_full_attention` (#553, merged).
- **zzf2024** — `h3_attention_o_gemm`, `h3_ffn_gate_up_gemm`, `h3_ffn_down_gemm` (strict block uses interim GEMM until then).
- **Jungle430** — QK norm, MM-RoPE, SwiGLU.
- **fxyan / Jerry536-coco** — sigma schedule / ODE step for scheduler parity.

We will keep **interim labels** in traces and docs until those rows land.

---

## 5. Open item: `stochastic_transition_contract`

Public H3 ODE sampler vs VeRL-Omni FlowGRPO CPS/SDE: draft in
[`stochastic-transition-contract-draft.md`](stochastic-transition-contract-draft.md).

**Ask:** approve probability model before we implement the RFC row and enable transition logprob comparators.

---

## 6. Phase 2 — VeRL-Omni (after RFC Diffusers gates)

Not a substitute for RFC merge. Intended as **external validation** of train/rollout consistency:

| Arm | Setup |
| --- | --- |
| P/P | Community fork `verl-omni` `main`, native Diffusers + vLLM-Omni |
| R/R | Fork branch `rl-kernel/h3`, `RL_KERNEL_H3_MODE=strict` both sides |

**Hypothesis (v0.1.0):** P/P shows non-zero logprob / latent drift (already ~1e-3 on tiny smoke); R/R shows **zero** drift with end-to-end time not worse than P/P (paired bootstrap, #377 method).

**Probe done:** CPU 270 tests; tiny FlowGRPO; TP=2 LoRA sync PASS — see [`probe-results.md`](probe-results.md).

**Needs:** community fork, full weights, integration wiring (#575 adapter + VeRL-Omni hooks).

Coordination docs: branch `plan/h3-verl-omni-e2e` on `fusheng-ji/RL-Kernel`.

---

## 7. Specific asks for maintainers

1. Confirm **merge order** #539 → #567 … #575 and whether #576 should wait.
2. Accept **correctness-first** merge with perf follow-ups, or require perf gate before `cuda_perf_closeout`.
3. Review **SDE contract draft** for `stochastic_transition_contract`.
4. Bless **Phase 2** VeRL-Omni experiment as community/partner track (optional co-marketing with verl-project).

---

## 8. Attachments checklist (before send)

- [ ] Table of PR links + latest green CI + `pr_assets` SHA
- [ ] H-C0–H-C7 matrix (updated)
- [ ] One-page perf chart (strict vs provider, stage breakdown)
- [ ] Dual trajectory summary (3 seeds)
- [ ] No credentials, no internal paths
