# Evidence of the `feat/h3-token-refiner` PR (staged, PR number not assigned yet)

Measurement outputs of branch `feat/h3-token-refiner` (head `b9e6b1c5b3` when staged), kept here instead of in the pull request.
Rename this directory to `pr<N>/` once the PR number is known. The structure below it is the repository path a runner writes to
(`reports/experiments/...`); only files this PR adds relative to its parent PR are included.

- Hardware / software: NVIDIA H100 (93 GiB, sm90), driver 580, CUDA 13.0, torch 2.13.0+cu130; pinned `MiniMax-H3@42ed227` weights.
- Commands: `tools/validation/models/h3_refiner_report.py --out reports/experiments/h3-token-refiner-h100/report.json --png .../figure.png`.
- Sanitization: host names were replaced by `node-a` / `node-b`; no user names, paths, job identifiers or site names remain in these files.
