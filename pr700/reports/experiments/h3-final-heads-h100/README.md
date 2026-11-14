# #700 at its head `3eda7d4f`: H100, 2026-11-14 (second run)

Measured on NVIDIA H100 (94 GB, sm90), 2 nodes x 4 GPUs, driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, Python 3.12, pinned `MiniMax-H3@42ed227` weights, diffusers `f53d5520`. The extension was built in place from the tested commit on that host. Paths are replaced by `<HOME>`, host names by `node-a` / `node-b`.

Each run folder holds `commit.txt`, `environment.txt` (GPUs, `_C` / `_H3_C` paths, `KERNEL_ALIGN_USE_FAST_MATH`), `pip-freeze.txt`, `status.txt` (exit code per stage) and the raw outputs: junit `tests.xml` and one `report.json` per run.

- `01-head/tests/`: `pytest tests/models/minimax_h3`, default build: 1122 / 3 / 0.
- `01-head/fulltrans/`: `h3_full_transformer.py`, strict (`full-transformer/`, `trace.boundary_bindings`: all eight modules `row`) and `--provider-boundaries`. Every compared value equals the run at `89cbd77f` (`../h3-strict-boundaries-h100/fulltrans/`) exactly.
- `01-head/dual/`: `h3_dual_trajectory.py`, seeds 11 / 22 / 33: strict 0/16, provider 16/16; equal to the run at `89cbd77f` exactly.
- `02-fast-math/`: `pytest tests/models/minimax_h3` with `KERNEL_ALIGN_USE_FAST_MATH=1` (and `RL_KERNEL_H3_STRICT=0`, which the build requires with fast math). The H3 extension `_H3_C` is not built in this mode, so its tests skip. `base-fm` is `test-h3` `07533aee` in the same build, the baseline for this row. `base-fm` 511 / 594 / 0, `pr700-fm` 531 / 594 / 0.

Each stage's stdout / stderr is next to its outputs as `<stage>.log` (added in a second commit). `01-head/mypy/mypy.log` is `mypy --ignore-missing-imports rl_engine/` in this environment, with torch installed: 36 errors in 12 files, the same set as `test-h3` `07533aee` with the same torch 2.13.0 and mypy 2.4.0. This PR adds none.
