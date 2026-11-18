# H100 evidence held back from the main results branch (2026-11-18)

These two PRs were split out of `handoff/h100-evidence-gaps-results-20261118` because their evidence is not complete: #536 has failures, #491 has no 8-GPU run. Setup, redaction and commands are as in that branch's `SUMMARY.md`; no patch was applied; failures are unedited and were not retried.

| PR | Head | Why held | Result |
|---|---|---|---|
| #536 | `4a83e4f8` | `test_det_gemm.py` and `tests/ops` fail | autotune suite 520 passed / 0 failed / 29 skipped. `test_det_gemm.py` and `tests/ops` fail after a CUDA illegal memory access in `test_target_shapes_invariance[shape3-cuda-deterministic_gemm]`, which is also the first failure on `upstream/main` `49941801` on the same node and build. Later failures in the same process are the sticky error, so the counts (816 failed on the head, 332 on main for `tests/ops`) are cascade sizes. The SM90 GEMM build flags were not tried. |
| #491 | `5882864c` | no 8-GPU run | 94 passed / 0 failed / 6 skipped (3 need eight CUDA GPUs, 3 need eight Ascend NPUs). Back-to-back benchmark vs `upstream/main` `49941801` on 2 and 4 GPUs only; nodes have 4 GPUs, and a two-node run would measure the inter-node network. |

Details, tables and the drafted "H100" paragraphs: `pr536/4a83e4f8/RESULTS.md`, `pr491/5882864c/RESULTS.md`.
