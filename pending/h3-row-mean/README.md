# h3-row-mean-v1: fixed-order row mean for the H3 CPS log-prob (2026-11-20)

Branch `perf/h3-fixed-order-row-mean` at `23ea31cc` (on `test-h3` `12e81c36`). Local paths are replaced by `<head>`, `<base>`, `<env>`, `<venv>`, `<verl-omni>` and `<python>`.

| Path | What |
|---|---|
| `row-mean-*.png`, `plot_row_mean.py` | the two charts of the PR's Results, drawn from the reports below |
| `row-mean/` | `fixed_order_mean` before (fixed-order GEMM) and after (`h3-row-mean-v1`): time per call, error against FP64, batch invariance; one B200 node allocated to the job alone, torch 2.13.0+cu130 |
| `logprob-vs-verl-omni/` | the CPS log-prob with the old and the new mean, against FP64 and verl-omni `a634f4d8`'s `sample_h3_transition`; batch invariance; torch 2.14.0+cu130, Diffusers 0.43.0 |
| `prior-art/` | vLLM 0.33's `batch_invariant.mean_dim` on the same rows: time (on a shared GPU), error, batch invariance, rows at another memory offset |
| `tests/suite-*.txt`, `failed-*.txt` | `pytest tests/models/minimax_h3 tests/backends/shared`, head on GPUs 0,1 and `test-h3` on GPUs 2,3 at the same time (`commits.txt`, `gpus.txt`) |
| `tests/runtime-integration-transition-venv.txt` | `test_h3_runtime_integration.py::TestTransition` with verl-omni importable |
