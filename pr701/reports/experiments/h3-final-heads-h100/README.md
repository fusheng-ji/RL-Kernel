# #701 at its head `e42c763a`: H100, 2026-11-14 (second run)

Measured on NVIDIA H100 (94 GB, sm90), 2 nodes x 4 GPUs, driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, Python 3.12, pinned `MiniMax-H3@42ed227` weights, diffusers `f53d5520`. The extension was built in place from the tested commit on that host. Paths are replaced by `<HOME>`, host names by `node-a` / `node-b`.

Each run folder holds `commit.txt`, `environment.txt` (GPUs, `_C` / `_H3_C` paths, `KERNEL_ALIGN_USE_FAST_MATH`), `pip-freeze.txt`, `status.txt` (exit code per stage) and the raw outputs: junit `tests.xml` and one `report.json` per run.

The grid folders hold one `report/report.json` (written by `tools/validation/models/h3_parallel_full_model.py`) and one `rankN/` folder per rank.

- `01-head/tests/`: `pytest tests/models/minimax_h3`, default build: 1187 / 18 / 0.
- `01-head/tp4-cp2-ring/` (size 40), `01-head/tp8-928/`: `all_bitwise` true, 350 weight shards compared and 0 mismatched on every rank.
- `01-head/tp4-cp2-ring-sp-inject/`: `--inject 17:16:3`; `all_bitwise` false, `first_drift_forward` block 17, row 16, owner rank 3 on every rank.
- `03-extra-layouts/`: TP4 x CP2 Ulysses at 40 / 925 / 928 + SP, and TP1 x CP3 ring at 40 (CP shards 14 / 13 / 13) and 925 (309 / 308 / 308): `all_bitwise` true, 350 shards, on every rank. `tp1-cp3-ulysses/` exited with status 1 on every rank and wrote no report; Ulysses needs `heads % cp == 0` and H3 has 56 heads.
- `02-fast-math/`: `pytest tests/models/minimax_h3` with `KERNEL_ALIGN_USE_FAST_MATH=1` (and `RL_KERNEL_H3_STRICT=0`, which the build requires with fast math). The H3 extension `_H3_C` is not built in this mode, so its tests skip. `base-fm` is `test-h3` `07533aee` in the same build, the baseline for this row. `base-fm` 511 / 594 / 0, `pr701-fm` 564 / 641 / 0.
