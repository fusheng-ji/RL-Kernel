# H100 evidence for RL-Align/RL-Kernel#701

2 nodes x 4 NVIDIA H100 94 GB (sm90), driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, Python 3.12.
Pinned `MiniMax-H3@42ed227` weights. The extension was built on the host from the tested commit. Hosts are `node-a` /
`node-b`, paths are `<SRC>` / `<ENV>`.

## `memfix/`: the grids that pass

- **Grids** (`grids/<grid>/`), one per grid:
  - `report/report.json` (`all_bitwise`, per-rank `checks`, `first_drift_*`, 350 weight-shard digests compared, `readback`);
  - one `rank<r>/` folder per rank with `commit.txt` and `ws2.log`.

  `grids/<grid>.<node>.mem.csv` holds the `nvidia-smi` memory samples.

  Every grid ran at `8db2cd98`. The `tp4-cp2-ring-sp-inject` control ran at `2fe3acb2`.

| Grid | Size | `all_bitwise` |
|---|---|---|
| tp2, tp4, tp2-cp2-ring, tp2-cp2-ulysses | 40 | true |
| tp4-cp2-ring, tp4-cp2-ring-sp, tp8 | 40 | true |
| tp4-cp2-ring-925, tp8-925 | 925 | true |
| tp4-cp2-ring-sp-928, tp8-928 | 928 (1008 rows) | true |
| tp4-cp2-inject (`--inject 17:5:3`) | 40 | false: block 17, row 5, owners [0, 1, 2, 3] |
| tp4-cp2-ring-sp-inject (`--inject 17:16:3`) | 40 | false: block 17, row 16, owner [3] |
| tp4-cp2-ring-sp-925 | 925 | refused by design: SP needs the 463 rows of CP rank 0 divisible by tp=4 |

- **Tests:** `tests/` (`2fe3acb2`) and `tests-8db2cd98/` hold `pytest tests/models/minimax_h3`. Both give 1182 passed,
  18 skipped, 0 failed; 15 of the skips need 5–8 GPUs on one node.
- **mypy:** `mypy/` and `mypy-8db2cd98/`, CI style, both clean.

## `c1455a75/`: the branch before the memory commits

`build/`, `tests/` (1179 passed, 18 skipped, 0 failed) and `mypy/`. In `grids/<grid>/rank<r>/ws2.log`, every grid runs
out of memory in the canonical run, on every rank, at about 95 GiB used.
