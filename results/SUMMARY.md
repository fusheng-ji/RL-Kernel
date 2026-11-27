# miles H2 H100 probe (JOB_X)

Gate: 2026-11-27T19:32:36Z FAIL b,c,d

- **a** (TP1 0.55): engine start failed, hybrid state cache negative; train rc=124; no perf 0.
- **b** (TP1 0.80): engines OK; one perf 0; NCCL ALLTOALL watchdog timeout at seq 881; train rc=1; peaks ~94 GiB/GPU.
- **c,d** (TP2): SGLang `Inkling2 tensor parallelism requires NVLink multicast`; no perf 0.

No other settings tried; no offload (README §H2). H3–H7 not run.

See `extra/h2_b_nccl_info_warn.txt`, multicast tracebacks, `extra/job_steps_timeline.txt`, and GPU diag in `extra/gpu_topo_nvlink_fabric.txt` when present.

## Log size notes (round 1 H2 evidence)
- `train.log` files were copied verbatim from the run tree (not head/tail truncated).
- Files over 90 MiB are stored as `*.log.gz` in this results branch.
- sacct steps <JOBID>.36–.41 (≈5 s, 2 CPU, no GPU, 2026-11-27T18:17Z): read-only overlap checks during probe b; no GPU, no recipe change.

