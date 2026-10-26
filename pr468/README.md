# Evidence of PR #468

Measurement outputs of RL-Align/RL-Kernel#468 (branch `feat/cuda-qwen3-next-gated-rmsnorm`, head `910bee3087` when moved),
kept here instead of in the pull request. The directory structure below `pr468/` is the
repository path the files had in the branch; only files the PR adds or changes relative to its
parent are included (unchanged files live under the parent PR's directory).

No site-specific strings were found in these files.

## C4 rerun with each library's own `dweight` (2026-10-26)

Review of #468 found that the reuse check's swapped wrappers inherited rl-kernel's
`parameter_vjp_contributions_fp32`, so the C4 gate judged rl-kernel's `dweight` rows instead of
the library's. After the fix (`339bc3ff`) the gates were rerun on a B200 from a clean tree:
`docs/usage/evidence/qwen3-next-norm-reuse-b200/qwen3_next_rms_norm_gates.json` (replaced) and
`rms_norm_gated_gates.json` (new), both unedited. `rms_norm_gated.json` and its figure take their
`contract_gates` from the new run (see its `evidence_corrections`); its other measurements were
not rerun. No site-specific strings were found in the new files.
