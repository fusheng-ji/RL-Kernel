# `feat/h3-grid-training-step`: the LoRA training step on the H3 grid and the 8-step G6 run

Evidence for the PR on branch `feat/h3-grid-training-step` (RFC #420 `final_distributed_h3_run`). All files are redacted: home, data and environment paths, user, host, cluster and job identifiers are replaced by placeholders.

| Path | What |
|---|---|
| `b200/g6-pp/` | G6 P/P (stock verl-omni, `a634f4d8` with the fork's launchers at `e62c4bd`): 8 steps, TP4 x CP2 on 8 x B200, `gate-lines.txt` (one metrics line per step), `train.log`, `summary.txt` (peak memory per GPU), `smi.csv` |
| `h3-grid-g6-pp-vs-rr-b200.png`, `.json` | chart and numbers: step time per phase, mean of steps 2–8, P/P against R/R (R/R from `../h3-gemm-rows-integration/b200/g6-rr-after/`) |
| `plot_g6.py` | draws the chart from the two `gate-lines.txt` |

The other runs of this PR are in `../h3-gemm-rows-integration/`: strict G5 on 8 x B200 (`b200/g5/`) and on 2 x 4 H100 (`h100/H3-strict-g5/`), and G6 R/R with the GEMM rows (`b200/g6-rr-after/`) and before them (`b200/g6-rr-before/`).
