# H100 evidence for RL-Align/RL-Kernel#699

NVIDIA H100 94 GB (sm90), driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, Python 3.12. Pinned
`MiniMax-H3@42ed227` weights and diffusers `f53d5520`. `_C` / `_H3_C` were built on the host from the tested commit with
`RL_KERNEL_H3_STRICT=1`. Hosts are `node-a`, paths are `<SRC>` / `<ENV>`.

Every folder has `commit.txt` (the tested commit, clean tree), `environment.txt`, `pip-freeze.txt` and `status.txt`.

| Folder | Commit | What | Result |
|---|---|---|---|
| `tests-base/` | `07533aee` (`test-h3`) | `pytest tests/models/minimax_h3`, 4 GPUs visible | 1102 passed, 3 skipped, 0 failed |
| `tests-followup/` | `e3c4cbc9` | same | 1114 passed, 3 skipped, 0 failed |
| `mypy-base/`, `mypy/` | `07533aee`, `e3c4cbc9` | `ci-mypy.log` is `mypy --ignore-missing-imports rl_engine/` as CI runs it | clean on both |
| `fulltrans-base/` | `07533aee` | `h3_full_transformer.py`, strict and `--provider-boundaries` | strict fails at `proj_in`: `parameter gradients require canonical_backward_session` |
| `fulltrans/` | `e3c4cbc9` | same | strict fails the same way; provider-boundaries completes |
| `fulltrans-vsess/` | `a6c93313` | same | both complete, candidate and provider bitwise repeatable |
| `dual/` | `e3c4cbc9` | `h3_dual_trajectory.py`, seeds 11 / 22 / 33 | strict 0/16 differing, provider 16/16, on every seed |

`fulltrans-vsess/full-transformer/report.json`, strict at `a6c93313`:
- video rel 2.151e-03, audio rel 2.104e-02;
- input-gradient rel 0.226 / 0.230 / 0.345;
- parameter-gradient rel median 0.266 / max 0.696;
- peak 68.2 GiB.
