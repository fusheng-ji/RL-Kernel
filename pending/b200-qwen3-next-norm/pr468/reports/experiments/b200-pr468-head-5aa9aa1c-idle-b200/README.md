# B200 evidence: PR #468 at its head `5aa9aa1c`, idle GPU

PR #468, fork branch `feat/cuda-qwen3-next-gated-rmsnorm`, head `5aa9aa1c` (full `5aa9aa1cf446c11bd4374c57c9ab81e0fb7b59a7`), sm_100, labelled B200 only. One NVIDIA B200 with no other process on it, torch 2.13.0+cu130, clean tree (`git status` empty before and after, `environment.txt`), extension built in place and checked to load from the checkout. Runner: `pr468/b200-head-align/run.sh` on this branch; commands and exit codes in `commands.tsv` / `exitcodes.tsv` (11 of 11 exit 0).

Why: the earlier B200 set at the head (`b200-pr468-head-5aa9aa1c-b200`) covers the unit suite, `check_operator` and the C3/C4 gates only and ran on a shared GPU; the norm report and the two comparison reports were measured at `1e2eaec0`, five commits before the head. This directory runs all of them at the head.

| Run | Result |
|---|---|
| unit suite (five paths in the PR's Tests) | 307 passed, 0 failed, 73 skipped |
| `check_operator` `rms_norm` / `rms_norm_gated` / `qwen3_next_rms_norm` | `pass_rate=1.0000` each |
| C3/C4 forward and gradient gates, both ops | 0 failed; the 25 / 44 / 25 / 69 invariance lines are identical to the earlier head run |
| norm report (`report.json`), comparison reports (`*_reuse.json`) | `git_commit` = head, tree not dirty |

Against the reports at `1e2eaec0`, the only differing fields are timings, commit ids and paths (filtered by field name; the filter is crude).

Transformer Engine, Megatron-core, Liger and FLA are not installed, so those comparison rows are absent. The figures were drawn from the JSON reports with the repository's own plot functions outside the benchmark environment (matplotlib is not installed there).
