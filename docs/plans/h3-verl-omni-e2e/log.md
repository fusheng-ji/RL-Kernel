# Decision log

## 2026-10-23

- Created coordination branch `plan/h3-verl-omni-e2e` (docs only, no PR).
- Documented blockers B1–B7, RFC gap, experiment design (#377 pattern), SDE draft, VeRL-Omni hook outline.
- Stage A probe (pre-branch): VeRL-Omni `efb9dc7`, 270 CPU tests pass, 2×H100 tiny FlowGRPO 2 steps pass;
  P/P `logprob_abs_diff_max ≈ 1.3e-3` on tiny random weights.
- Weights: 1/14 HF transformer shards present; full download blocked until quota check (B1).
- Flash vs #553 evidence published locally; PR #576 opened separately from this branch.
- Pushed coordination branch to `fusheng-ji/RL-Kernel` @ `991e4882`.
- B1: `run_minimax_h3_lora_sync_tp2.sh` PASS on 2×H100; weights 1/14 (quota TBD).
- B4: `verl_omni/integrations/rl_kernel/` skeleton on local verl-omni `rl-kernel/h3`.
- B7: `examples/minimax_h3_verl_omni_e2e/experiment_matrix.json` scaffold added.
- User direction: **RFC closeout before maintainer proposal**; added `rfc-closeout-roadmap.md`,
  `RFC-TASKS.md` (R1–R8), `maintainer-proposal-outline.md` (draft, do not post yet).
- R3 weights: confirmed **14/14** shards already present under `hf_full/transformer` (pinned
  `MiniMaxAI/MiniMax-H3` @ `42ed227e`); shard-1 sha256 matches `h3_manifest.json`.
- Added CUDA ownership tables (unclaimed / contested / external deps) to `rfc-gap.md` for maintainer proposal.
- Audio projection overlap: co-author invite to walker-ai on PR #568 (documented in `rfc-gap.md` §B).
