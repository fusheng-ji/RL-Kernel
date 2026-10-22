# Evidence of the `feat/h3-token-refiner` PR (staged, PR number not assigned yet)

Measurement outputs of branch `feat/h3-token-refiner` (measured at head `b9e6b1c5b3`; current head `67af7966`, which adds row-permutation-invariant parameter gradients and does not change the forward path, so these numbers were not re-measured), kept here instead of in the pull request.
Rename this directory to `pr<N>/` once the PR number is known. The structure below it is the repository path a runner writes to
(`reports/experiments/...`); only files this PR adds relative to its parent PR are included.

- Hardware / software: NVIDIA H100 (93 GiB, sm90), driver 580, CUDA 13.0, torch 2.13.0+cu130; pinned `MiniMax-H3@42ed227` weights.
- Commands: `tools/validation/models/h3_refiner_report.py --out reports/experiments/h3-token-refiner-h100/report.json --png .../figure.png`.
- Supplement (2026-10-23): `reports/experiments/h3-token-refiner-h100/swiglu_tolerance_matrix.json` — all six `test_nodes_accuracy_repeat_batch_and_isolated` swiglu nodes on H100, plus a cross-device reference to the B200 `#570` failing case (`block=1`, `tokens=257`). Replay used GitHub PR #570 head `6de320dc` (`git fetch origin feat/h3-token-refiner`); the JSON documents that B200 body text uses parallel head `67af7966` and why both appear.
- Sanitization: host names were replaced by `node-a` / `node-b` where applicable; no user names, local paths, job identifiers or site names remain in these files.
