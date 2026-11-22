# `feat/h3-parallel-v2-gemm-rows`: `h3-parallel-v2`, the TP gate/up, O and down projections on the rows' kernel

Evidence for the PR on branch `feat/h3-parallel-v2-gemm-rows` (head `093b0ff2`, base `81a87617`). All files are redacted: home, data and environment paths, user, host, cluster and job identifiers are replaced by placeholders.

| Path | What |
|---|---|
| `h3-parallel-v2-crosscheck-h100.png` | chart: the 8-unit pieces against the rows' TP = 1 operators, max abs difference (from `h100/H2/crosscheck.json`) |
| `h3-parallel-v2-gemm-time-b200.png` | chart: per-rank GEMM time, v1 vs v2 kernel, TP shard shapes (from `b200/bench/bench.json`) |
| `h100/H2/crosscheck.json`, `crosscheck.log` | `tools/validation/models/h3_gemm_rows_crosscheck.py` on one H100 |
| `h100/H2/kernel-tests.txt` | `test_h3_fixed_fp32_gemm.py` on H100 |
| `h100/H2/<grid>/` | one directory per grid against `Grid(1, 1)` (2 nodes x 4 H100): `report.json`, `run.log`, command, commit, environment, tree status |
| `h100/H2/suite-head.txt`, `suite-base.txt` | `tests/models/minimax_h3` at head and base |
| `b200/suite/` | B200 suites at head and base, the 8-GPU files (`pr2-8gpu.txt`), environment |
| `b200/bench/` | `h3_gemm_rows_bench.py` and its `bench.json` (one B200; also times the pack and noise-mix rows of `feat/h3-pack-noise-rows`), commit, GPU |

The integration runs (strict G5 on H100 and B200, G6 R/R on B200) are in `../h3-gemm-rows-integration/`.
