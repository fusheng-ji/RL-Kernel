# Evidence of PR #574 (`feat/h3-runtime-integration`)

Measurement outputs, kept here instead of in the pull request.

- Hardware / software: NVIDIA H100 (93 GiB, sm90), CUDA 13.0, torch 2.13.0+cu130; pinned `MiniMax-H3@42ed227` weights; verl-omni scheduler loaded from a checkout.
- Command: `tools/validation/models/h3_runtime_report.py --out reports/experiments/h3-runtime-integration-h100/report.json --png .../figure.png --checkpoint <dir> --verl-omni <checkout>`.
- The 8349-token strict / audit rows were run with the `h3-flash-attention-v1` binding (an O(n) fixed-order attention that is not part of this PR's branch); with this branch's interim `DeterministicAttentionOp` the same rows run out of memory (recorded in the report). Command for that run: `h3_runtime_report.py --sizes 8349 --no-ablations ...` on the stack that has the flash row.
