# H100 evidence for RL-Align/RL-Kernel#700

NVIDIA H100 94 GB (sm90), driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, Python 3.12. Pinned
`MiniMax-H3@42ed227` weights and diffusers `f53d5520`. `_C` / `_H3_C` were built on the host from the tested commit with
`RL_KERNEL_H3_STRICT=1`. Hosts are `node-a`, paths are `<SRC>` / `<ENV>`.

All runs are at `89cbd77f`. Every folder has `commit.txt` (clean tree), `environment.txt`, `pip-freeze.txt` and `status.txt`.

| Folder | What | Result |
|---|---|---|
| `tests/` | `pytest tests/models/minimax_h3`, 4 GPUs visible | 1118 passed, 3 skipped, 0 failed |
| `mypy/` | `mypy --ignore-missing-imports rl_engine/` as CI runs it | clean |
| `fulltrans/` | `h3_full_transformer.py`, strict and `--provider-boundaries` | all 8 boundary modules on `row` in strict mode; candidate and provider bitwise repeatable |
| `dual/` | `h3_dual_trajectory.py`, seeds 11 / 22 / 33 | strict 0/16 differing, provider 16/16, on every seed |
| `build/` | the extension build log | |

`fulltrans/full-transformer/report.json`, strict:
- video rel 2.511e-03, audio rel 1.543e-02;
- input-gradient rel 0.118 / 0.156 / 0.248;
- parameter-gradient rel median 0.176 / max 0.434;
- peak 68.2 GiB.
