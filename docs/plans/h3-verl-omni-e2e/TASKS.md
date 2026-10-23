# Task tracker (source of truth)

Update **Status** and **Evidence** here before starting work on a blocker. Status values:
`pending` | `in_progress` | `blocked` | `done`.

## Summary

| ID | Title | Status | Depends on |
| --- | --- | --- | --- |
| B1 | Probe: full weights + native VeRL-Omni baseline | blocked | — |
| B2 | SDE transition contract draft | done | — |
| B3 | Performance: GEMM tiles, flash, linears | in_progress | — |
| B4 | VeRL-Omni integration branch | in_progress | B1, B2 |
| B5 | RFC gap table and final rows | done | — |
| B6 | Tiny-scale R/R smoke | blocked | B4 |
| B7 | Full experiment matrix + result bundle | in_progress | B3, B4, B5, B6 |

---

## B1 — Probe: weights and native baseline

**Goal:** Confirm VeRL-Omni H3 train+rollout on real (or staged) weights; record P/P inconsistency.

**Done when:**

- All 14 transformer shards present under the pinned HF layout (or documented quota blocker).
- Native run succeeds: tiny smoke (done), plus rollout **TP=2** script path exercised on GPU.
- P/P metrics logged: `rollout_corr/logprob_abs_diff_max` and mean at ≥1 step on real weights.

**Current:**

- Tiny random weights: 270 CPU tests passed; 2×H100 tiny FlowGRPO T2VA 2 steps passed.
- P/P on tiny: `logprob_abs_diff_max ≈ 1.3e-3` (non-zero).
- Weights: 1/14 shards on disk (~4.5 GB); full ~134 GiB bf16 not yet local.

**Evidence:**

- Internal handoff note `stage-a-probe-notes.md` (not in repo; summarized in `log.md`).
- Repro scripts: VeRL-Omni `tests/special_e2e/run_flowgrpo_minimax_h3_tiny.py`,
  `tests/special_e2e/run_minimax_h3_lora_sync_tp2.sh`.

**Next actions:** quota check → HF download remaining shards → P/P metrics on real weights.

**Update 2026-10-23:** TP=2 LoRA sync regression **PASS** — see [`probe-results.md`](probe-results.md).
Weights still 1/14 (blocked on quota).

---

## B2 — SDE transition contract

**Goal:** Maintainer-approved probability model for FlowGRPO CPS/SDE steps (video + audio schedulers).

**Done when:**

- Draft merged in this plan folder; issue #420 comment posted **after user approval**.
- RL-Kernel row `stochastic_transition_contract` implementation PR opened (separate from this branch).

**Current:** Draft in [`stochastic-transition-contract-draft.md`](stochastic-transition-contract-draft.md).

**Depends on:** VeRL-Omni `FlowMatchSDEDiscreteScheduler.sample_previous_step`, `sde_type=cps`,
`noise_level=0.8`.

---

## B3 — Performance closeout

**Goal:** Strict H3 step time within reach of provider on dominant shapes (925–8349 tokens).

**Done when:**

- Shape-based projection GEMM tile pick in a PR (from local `h3-projection-gemm` work).
- Optional flash backend (#576) available in stack; block no longer attention-dominated.
- Documented H100 evidence: strict/provider ratio trend toward ≤2× on full step (interim target).

**Current:** #553 strict attention 11–35× vs provider on stack; flash ~2× e2e at 925 in local measure;
`o_proj` / `ffn_gate_up` / `ffn_down` interim GEMM dominate block time.

**Evidence:** PR [#576](https://github.com/RL-Align/RL-Kernel/pull/576); `pr_assets` flash-vs-553 JSON.

**Update:** shape-based tile pick lives on local `local/h3-step-time` (`c75fa192`); open PR after #568 merges
(cherry-pick onto `test-h3` alone is insufficient — no `h3_projection.py` on upstream yet).

---

## B4 — VeRL-Omni integration

**Goal:** Fork branch `rl-kernel/h3` routes train and rollout to RL-Kernel H3 strict ops with readback.

**Done when:**

- `verl_omni/integrations/rl_kernel/` (or agreed path) with `RL_KERNEL_*` env routing.
- Rollout: vLLM-Omni H3 layers replaced per [`verl-omni-integration-outline.md`](verl-omni-integration-outline.md).
- Actor: Diffusers block forward/backward replaced before FSDP wrap.
- Fused QKV ≡ split q/k/v column test; Ulysses SP aligned with rollout.
- Bitwise comparators for velocity / `prev_sample_mean` wired into metrics.

**Update 2026-10-23:** Skeleton package `verl_omni/integrations/rl_kernel/` on local `rl-kernel/h3` branch
(manifest, metrics, actor/rollout patch stubs). Patches raise `NotImplementedError` until #575 adapter is wired.

**Blocked on:** push to community `verl-omni` fork; #575 merged or manifest-pinned local build.

**Outline:** [`verl-omni-integration-outline.md`](verl-omni-integration-outline.md).

---

## B5 — RFC gap

**Goal:** Close documentation and experiment gaps vs [#420](https://github.com/RL-Align/RL-Kernel/issues/420).

**Done when:**

- [`rfc-gap.md`](rfc-gap.md) complete and kept current.
- R/P and P/R arms executed at least once before final closeout.
- Rows `final_e2e_diffusers_cuda`, `final_distributed_h3_run`, `cuda_perf_closeout` claimed **after user approval**.

---

## B6 — Tiny R/R smoke

**Goal:** 2×GPU tiny weights, R/R arm: zero drift every step; P/P non-zero.

**Done when:** Same metric pipeline as #377 on ≥8 steps; logged CSV row per step.

**Depends on:** B4 patches implemented (not stubs). **Blocked** until actor/rollout hooks call RL-Kernel.

---

## B7 — Full matrix

**Goal:** Matched-workload P/P vs R/R (+ R/P, P/R for localization), 3 seeds, short then long horizon.

**Done when:**

- `experiment_matrix.json`, `rounds.csv`, `summary.json`, plots (pattern from `examples/vime_qwen3_8b_tp4_cp2_200/`).
- 8×H100, real weights, rollout TP=2, actor FSDP + Ulysses SP as in VeRL-Omni H3 recipe.

**Depends on:** B3 sufficient rollout perf, B6 green, B1 weights, Slurm 8-GPU job.

**Update 2026-10-23:** Scaffold `examples/minimax_h3_verl_omni_e2e/experiment_matrix.json` and README on this
branch; analyze/collect/plot scripts TODO (copy from vime example when B6 runs).

---

## Risks (standing)

| Risk | Mitigation |
| --- | --- |
| PR stack not on `test-h3` | Pin manifest SHAs; rebase integration as PRs merge |
| No 8-GPU single node | 2×4 hierarchical collective (#569); WS2 tests skip on 4-GPU |
| SDE contract not approved | Report velocity/latent drift only until approved |
| Native rollout fused ≠ Diffusers split | Column-equivalence tests in B4 |
| Disk quota | Check before full weight download |
