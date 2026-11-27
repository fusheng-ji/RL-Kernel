# Partial run summary (schedule push after redact fix `2c49e44b`; JOB_X RUNNING, JOB_Y PENDING)

## Pipeline deviations (2026-11-27, jobx B1)

- **P2 first attempt** (`pipeline_P2_jobx_attempt1_no_MASTER_ADDR`): rows not all exit=0 — missing `MASTER_ADDR`/`MASTER_PORT` in nested srun. **Accepted rerun** (runner-only): `MASTER_ADDR` from job node-a, `MASTER_PORT` per row, rank-0 driver `srun -w` pinned to node-a. **Not a code/recipe change.**
- **Second attempt P2**: **9/9 `exit=0`** in `partial-results/pipeline/P2/rows.txt`.
- **P3/P4 first attempt** (`pipeline_P3P4_jobx_attempt1_single_node_srun`): 5+5 rows `exit=1`, `Only allocated 1 nodes asked for 2` — chain should have STOPped; fixed `u_pipeline.sh` to exit 1 when any row is non-zero; P3/P4 now use `driver_2node_sync` + `slurm_job_env_exports`.
- **jobxB1b (step .55`)**: continued into vime V2 after bad P3/P4; V2 FAIL (`smoke--num-layers_4` rc=1 on shared card). V3/V4 not run. `START=P3` relaunch runs **P3/P4 only** then exits.

# Partial run summary (second incremental push; JOB_X still RUNNING, JOB_Y PENDING)

**Allocation:** two 2-node x 4 H100 jobs, non-exclusive, 64 CPU per node. A ran on JOB_X after step0 retries. `nodea_done` and `nodeb_done` are both set. B2a (miles H2) is running; B2b, C and B1 have not started (JOB_Y pending).

## Deviations (PLAN section 0)

Not `--exclusive`; 64 CPU; four fixed CUDA lanes on node-b; vime V0 runs as a separate single-card step with 32 CPUs. Scheduler-side fixes are listed at the end; attempts are kept in `not-green-attempts/`.

## Results by slice

Verdicts are read from logs and exit codes; they are not a re-judgement of the handoff pass rules.

| Slice | Where | Status |
|---|---|---|
| step0, nested srun, builds W10/WB/WP/WQ | `checks/`, `partial-results/build/` | OK |
| miles H0, H1 | `partial-results/miles/` | H0 digest and 16 sha equal; H1 done |
| T10-a | `partial-results/T10/a/` | suite 535 passed; capture, run-edit, run-t2i, run-b, t2i-vs-t9 all exit 0 |
| T10-edit | `partial-results/T10/edit/` | exit 0, tree clean |
| QK2 | `partial-results/QK2/` | build exit 0; real-weights QK tests 2 passed; **ws1 candidate evidence FAILED on head (WQ) and base (WB)**: `det_gemm` case `gemm-short-m8-k4096-n4096` forward_accuracy false, max_abs_error 0.5, cuBLASLt no-split-K startup contract unavailable (fallback). Kept as a result, not changed |
| vime V0 | `partial-results/vime/` | **OK on the third attempt** with handoff `cd4fda86` (pip pinned to the B200 freeze); freeze diff empty |
| vime V2 | `not-green-attempts/vime_V2_attempt1_shared_gpu/` | **gate was set OK by mistake** (smoke.sbatch exit 0) although `smoke--num-layers_4` and `rollout_smoke` returned 1 (CUDA OOM: GPU 0 already held ~13.9 GiB by another step). Gate reset to FAIL; V2 is rerun on an idle card at the start of B1 |
| verl-omni F0-F2 | `partial-results/verlomni/` | F0 gate OK; all 6 w0 runs `train exit=0` |
| verl-omni F3 (1344x768, 107 frames) | `partial-results/verlomni/runs/w1-*` | both arms OOM in `update_weights_from_ipc` (GPU 1 about 95 GB; node-a GPUs were idle at start). README: an OOM is a result. The monitor shows `verlomni_F3=pending` only because the script never wrote that gate. Rerun on an idle job is queued |
| pipeline P1 | `not-green-attempts/pipeline_P1P2_*` (after rerun starts) and `partial-results/pipeline/P1` | 7 failed, 205 passed, 4 skipped (3 + 1 need 8 GPUs). Five failures seen are `CUDA error: invalid argument` in `DeterministicCollective` IPC. Cause found on our side: the schedule script exported `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `rl_engine._C.deterministic_collective_ipc_meta` fails with it and works without (single-card repro). Not set by the handoff README |
| pipeline P2 | `partial-results/pipeline/P2` | all 9 rows exit 1 with `srun: More processors requested than permitted` (the srun inherited 64 CPUs per task from the node step). Cause inferred from the step settings; not reproduced separately |
| miles H2 | `partial-results/miles/h2_memprobe.sh-*.log` | gate not written yet. a (TP1, 0.55): SGLang cannot start (hybrid state cache memory negative), 503 retries until timeout, rc 124. b (TP1, 0.80): engines up, weights updated, rollout done (`perf 0` once), then NCCL ALLTOALL watchdog timeout after 600 s, rc 1; peak about 94 GB per GPU. c (TP2, 0.55): `Inkling2 tensor parallelism requires NVLink multicast`, no `perf 0`. d (TP2, 0.70): same error so far, waiting for its timeout |

## Failures kept as-is (`not-green-attempts/`)

1. vime V0 attempt 1: unconstrained pip pulled flash-attn-4 and cutlass-dsl 4.8.1 (the failure behind the handoff update).
2. vime V0 attempt 2 (handoff `cd4fda86`): `STEP_FAIL flashmla`, `freeze_differs_from_b200`, `flash-attn-4_installed`. Cause: enroot mounted the host home as `/root`, so user-site packages and an earlier clone were visible in the container before any pip step. Fixed in the runner with `ENROOT_MOUNT_HOME=n`; handoff files unchanged.
3. vime V2 attempt 1 (above).
4. Early A step0: env permission error, nested verdict parser; node-b restarts (lane cpuset ranges, missing `out/vime`).

## Reruns queued on JOB_X (start after B2a/B2b finish or miles_H2 is FAIL)

P1, P2 (with `expandable_segments` removed and srun `-c` set), then verl-omni F3, on an idle node-a, 8 h step. Old P1/P2/F3 outputs move to `not-green-attempts/` first. B1 will rerun vime V2 before V3/V4.

## Script changes since the previous push

`u_vime.sh` (`ENROOT_MOUNT_HOME=n`), `u_vime_retry.sh`, `alloc_B1.sh` (V2 rerun before V3), `u_pipeline.sh` (no `expandable_segments`, srun `-c`), `u_f3_rerun.sh`, `f3_rerun_wait.sh`, `imgcheck.sh`, `ipc_repro.py`, `stage_push.sh` (also stages `not-green-attempts` from the run tree).
