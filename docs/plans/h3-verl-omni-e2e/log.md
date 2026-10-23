# Decision log

## 2026-10-23

- Created coordination branch `plan/h3-verl-omni-e2e` (docs only, no PR).
- Documented blockers B1–B7, RFC gap, experiment design (#377 pattern), SDE draft, VeRL-Omni hook outline.
- Stage A probe (pre-branch): VeRL-Omni `efb9dc7`, 270 CPU tests pass, 2×H100 tiny FlowGRPO 2 steps pass;
  P/P `logprob_abs_diff_max ≈ 1.3e-3` on tiny random weights.
- Weights: 1/14 HF transformer shards present; full download blocked until quota check (B1).
- Flash vs #553 evidence published locally; PR #576 opened separately from this branch.
