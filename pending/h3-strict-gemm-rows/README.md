# `feat/h3-strict-gemm-rows`: the O, gate/up and down GEMM rows in the strict WS1 graphs

Evidence for the PR on branch `feat/h3-strict-gemm-rows` (head `02a0ef34`, base `9388feed`). All files are redacted: home, data and environment paths, user, host, cluster and job identifiers are replaced by placeholders.

| Path | What |
|---|---|
| `h3-strict-gemm-rows-h100-error.png` | chart: forward error of the three GEMM nodes against the FP32 golden, operator vs Diffusers (from `h100/H1/block-replay.json`) |
| `h100/H1/` | H100 at `775e98e7`: `block-replay.json` / `.log` (block 0, S = 264 / 925 / 3160, forward and backward), `refiner-report.json` / `.log`, `binding-tests.txt` (the run with the one head-only failure), `suite-head.txt` / `suite-base.txt`, commit, environment, `nvidia-smi`, tree status before and after each step |
| `h100/H1R-rerun/` | H100 rerun with `02a0ef34` applied as a patch on `775e98e7` (commit `f6d4ee71`, same tree `c6bf8c3e`): `binding-tests.txt` (2 files, 33 passed), `binding-tests-readme.txt` (4 files, 96 passed), `suite-head.txt` (1475 passed), the test IDs collected at head and base and the 5 present only at head, the patch, the step log |
| `b200/suite/` | B200 at `775e98e7` and base `9388feed`: suite outputs, the failing test lists, environment, GPUs |
