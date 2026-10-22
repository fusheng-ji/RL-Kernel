# Evidence of the `feat/h3-full-transformer` PR (staged, PR number not assigned yet)

Measurement outputs of branch `feat/h3-full-transformer` (measured at head `c3cd0358d9`; current head `e4524471`, which adds row-permutation-invariant parameter gradients and does not change the forward path, so these numbers were not re-measured), kept here instead of in the pull request.
Rename this directory to `pr<N>/` once the PR number is known. The structure below it is the repository path a runner writes to
(`reports/experiments/...`); only files this PR adds relative to its parent PR are included.

- Hardware / software: NVIDIA H100 (93 GiB, sm90), driver 580, CUDA 13.0, torch 2.13.0+cu130; pinned `MiniMax-H3@42ed227` weights.
- Commands: `tools/validation/models/h3_full_transformer.py and h3_dual_trajectory.py (seeds 11, 22, 33, with and without --diag-video-projections); tools/validation/models/h3_full_charts.py <dir> figure.png`.
- Sanitization: host names were replaced by `node-a` / `node-b`; no user names, paths, job identifiers or site names remain in these files.
