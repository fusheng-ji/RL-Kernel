# H100 results: Qwen3-Next rollout re-measure at the CUDA default connection limit (3e06906f, handoff 9b7413e7)

Hardware: one node with 4 x NVIDIA H100 (95830 MiB, driver 580.173.02).
- The node was exclusive for the whole run: `squeue` on it showed only this job at the start and at the end (`squeue-node-start.txt`, `squeue-node-end.txt`), and no other GPU process ran before A.
- Other nodes of the same allocation ran unrelated single-GPU tasks during C; they share no GPU, no `/dev/shm` and no host with this node.

Code: `3e06906f` comes from the handoff bundle, which was verified (`git bundle verify`, prerequisite `8fbacb87`) and fetched as `local/rollout-default`. The tracked status was empty before and after every step.

Environment: last round's training environment, unchanged and with nothing installed: torch 2.13.0+cu130, vLLM 0.30.0, flash-linear-attention 0.4.2, fla-core 0.4.2 (see `*/header.txt`). Also `PYTHONNOUSERSITE=1` and `CUDA_DEVICE_MAX_CONNECTIONS` unset in the shell.

| Step | Command (README) | Exit | Result |
|---|---|---|---|
| A build | `python setup.py build_ext --inplace` | 0 | `c-file.txt` path inside the checkout; status empty |
| B chain | `qwen3_next_run.py --stage all --until l2` | 0 | `preflight`, `layers`, `l2` all `passed`, each with `source_commit` `3e06906f…` |
| C bench | `qwen3_next_e2e_bench.py --run <l2> --phases rollout` | 0 | `rl_kernel_commit` `3e06906f…`, `tracked_tree_dirty: false`; `unsupported` lists only `native_bi` for both workloads; every strict shape has its three checks true (`row0_equals_batch1` null at batch 1) |

## Rollout totals (s): this round (3e06906f) / last round (37e04258)

| Workload | Shape | strict | stock, CUDA graphs (`native_graph`) | stock, eager (`native_eager`) |
|---|---|---:|---:|---:|
| certified | p64 b1 | 4.54 / 4.52 | 0.100 / 0.100 | 1.77 / 1.75 |
| certified | p64 b4 | 5.17 / 5.16 | 0.236 / 0.238 | 2.03 / 1.99 |
| certified | p256 b1 | 4.55 / 4.53 | 0.182 / 0.183 | 1.77 / 1.74 |
| certified | p256 b4 | 5.23 / 5.21 | 0.311 / 0.318 | 2.02 / 1.99 |
| certified | p1024 b1 | 4.70 / 4.68 | 0.174 / 0.174 | 1.76 / 1.73 |
| certified | p1024 b4 | 5.61 / 5.57 | 0.308 / 0.313 | 2.01 / 1.97 |
| long | p512 b1 | 163.0 / 163.2 | 2.27 / 2.26 | 55.66 / 54.85 |
| long | p512 b16 | 217.0 / 217.2 | 3.50 / 3.50 | 56.90 / 56.25 |
| long | p512 b64 | 373.4 / 370.5 | 5.93 / 5.93 | 57.80 / 57.06 |

**Finding: dropping `CUDA_DEVICE_MAX_CONNECTIONS=1` for the stock arms changes almost nothing on H100.**
- `native_graph` totals are 0.98–1.00x last round's.
- `native_eager` totals are 1.01–1.02x last round's, that is 1.2–1.9% slower, not faster.
- Strict, whose environment is unchanged, is 1.00–1.01x. So the run-to-run spread between the two rounds is about 1%, and the eager change is barely above it.
- The strict-over-stock ratios therefore stay where they were: 16–72x over `native_graph` and 2.5–6.5x over `native_eager` (`report.json` `strict_over_native`).
- `engine_arguments`, `settings`, `versions` and `topology` in the report equal last round's.
- The report does not record each arm's environment. That the stock arms ran without the variable follows from the code at `3e06906f` (`arm_environment` drops it for `native_graph`/`native_eager`, and its new test asserts this), not from a field in the report.

No failed attempts: nothing was retried, so there is no `not-green-attempts/`.
