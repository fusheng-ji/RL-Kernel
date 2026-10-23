# RFC #420 task tracker (R1–R8)

Status: `pending` | `in_progress` | `blocked` | `done`.  
Parent plan: [`rfc-closeout-roadmap.md`](rfc-closeout-roadmap.md).

| ID | Task | Status | Depends | Evidence / next step |
| --- | --- | --- | --- | --- |
| R1 | PR stack #539,#567–#575 CI/DCO; respond review | in_progress | — | Per-PR checks; rebase on `test-h3` as needed |
| R2 | Refresh `pr_assets` + PR bodies (SHAs, figures) | pending | R1 | `pending/h3-*` dirs; no local paths in public text |
| R3 | Full weights; rerun #571 33B + dual trajectory | blocked | quota | 1/14 shards; see probe-results |
| R4 | H-C0–H-C7 matrix (pass/gap/N/A) | in_progress | R3 partial | [`rfc-gap.md`](rfc-gap.md); fill H-C6 |
| R5 | §6.4 P/P,R/R,R/P,P/R on Diffusers strict stack | pending | R3, #574 merged | Extend `h3_dual_trajectory` runners |
| R6 | H1–H17 ablation signature table | pending | #574 | Overlay exists; need published mismatch runs |
| R7 | PR for `final_e2e_*` + `cuda_perf_closeout` rows | pending | R3–R6, perf story | Single or split PRs; user approval to claim |
| R8 | Finalize maintainer proposal (English) | pending | R1–R7 credible | [`maintainer-proposal-outline.md`](maintainer-proposal-outline.md) |

## Immediate priorities (RFC-first)

1. **R1** — unblocking review (docs links, 8-GPU help wanted text, DCO).
2. **R3** — weights (ask admin quota; download 14 shards).
3. **R4** — H-C6 grad-checkpoint test plan (add to #571 or closeout PR).
4. **Interim perf narrative** — document zzf2024 row dependency for `cuda_perf_closeout` (R7).

VeRL-Omni tasks B1–B7 remain in [`TASKS.md`](TASKS.md); do not prioritize B4–B7 until R5/R7 Diffusers path is credible.
