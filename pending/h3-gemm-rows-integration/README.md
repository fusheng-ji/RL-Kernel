# The GEMM rows in the multi-step run (`feat/h3-grid-training-step`)

The training-step branch merged with `feat/h3-strict-gemm-rows` (`775e98e7`), `feat/h3-parallel-v2-gemm-rows` (`a63afbdb`; its head `093b0ff2` adds only the SM90 cross-check tool) and `feat/h3-pack-noise-rows` (`2c9adb7e`), plus #734's build fix. Each run's `commits.txt` / `commit.txt` names the tree it ran. All files are redacted: home, data and environment paths, user, host, cluster and job identifiers are replaced by placeholders.

| Path | What |
|---|---|
| `h3-gemm-rows-g6-step-time-b200.png`, `g6-step-time.json` | chart and numbers: G6 R/R step time, mean of steps 2–8, before (v1 GEMMs) and after |
| `b200/g5/` | strict G5, TP4 x CP2 on 8 x B200: `report.json` and `run.log` per case (random / zero / clipped / recipe size / Ulysses / injected error), the rollout engine check, `summary.txt` |
| `b200/g6-rr-after/` | G6 R/R with the GEMM rows: `gate-lines.txt` (one metrics line per step), `train.log`, `summary.txt` (peak memory per GPU), `smi.csv`, commits |
| `b200/g6-rr-before/` | the same run before the GEMM rows (v1), for the comparison |
| `h100/H3-strict-g5/` | strict G5 on 2 nodes x 4 H100, the same tree as `b200/g5/`: one directory per case with `report.json`, `run.log`, command, commit, environment |
| `scripts/plot_gemm_rows.py` | draws the charts of this folder and of the two PR folders |
