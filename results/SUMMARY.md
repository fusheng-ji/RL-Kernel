# miles H2 H100 probe (JOB_X)

Gate: FAIL b3 (round 3, 2026-11-28); round 1: 2026-11-27T19:32:36Z FAIL b,c,d

- **a** (TP1 0.55): engine start failed, hybrid state cache negative; train rc=124; no perf 0.
- **b** (TP1 0.80): engines OK; one perf 0; NCCL ALLTOALL watchdog timeout at seq 881; train rc=1; peaks ~94 GiB/GPU.
- **c,d** (TP2): SGLang `Inkling2 tensor parallelism requires NVLink multicast`; no perf 0.
- **b3** (round 3, TP1 0.80, same setting as b, with hang diagnostics): **FAIL, same hang as b.** Rollout done (`perf 0:`), no trainer ` - step 0: {` line, train rc=1. Details below.

No other settings tried; no offload (README §H2). H2 failed, so H3–H7 not run (README: stop and report).

## H2 probe b3 (round 3, 2026-11-28)

Round-1 outputs were moved to `not-green-attempts/round1/` before this run. b3 is the single round-3 probe (README §H2), with `TORCH_NCCL_DUMP_ON_TIMEOUT` flight-recorder dumps and py-spy stacks taken during the hang.

**Pre-run checks (scheduler side):** `gpu_check_node{0,1}.txt`: `nproc 64` on both nodes; `nvidia-smi-start-node*.txt`: 0 MiB used on all 8 GPUs; no other job steps on the GPUs during the probe.

**Timeline (UTC; `train.log` and `gpu_mem_*.csv` timestamps are CET = UTC+1):**
- 00:32Z start; 00:44Z rollout peak 93.5 GiB/GPU (node-a) and 90.9 GiB/GPU (node-b), from `gpu_mem_node*.csv`.
- 00:47:14Z rollout `perf 0:` (256 samples, `perf/rollout_time` 413.6 s).
- 00:47:31Z all 8 trainer ranks enter collective **seq 881**: `nccl:all_to_all`, 8 × 16 elements, the MoE token-count exchange in the first `log_probs` forward (`_token_count_exchange` token_dispatcher.py:276 ← `_dispatch`:490). Seq 880 (`_all_gather_base`) completed on all ranks. GPU memory stays ~30 GiB/GPU from here on.
- 00:57:31Z rank 0 watchdog: `WorkNCCL(SeqNum=881, OpType=ALLTOALL, NumelIn=128, NumelOut=128, Timeout(ms)=600000)`; 00:58:31Z rank 0 aborts, train rc=1.

**Where each rank was during the hang:**
- **Rank 0** finished enqueueing seq 881 (watchdog: `Last enqueued NCCL work: 881, last completed NCCL work: 880`) and waits in `_sync_token_count_exchange` (token_dispatcher.py:313), `stacks/node0_r2_pid*.txt`. Flight recorder: seq 881 `state=scheduled`.
- **Ranks 1–7** never finished enqueueing seq 881 (watchdog: `Last enqueued NCCL work: 880, last completed NCCL work: 880`). Their py-spy stacks sit in `all_to_all` ← `_token_count_exchange`:276, and their NCCL logs show the launch stuck in lazy P2P connection setup: `ncclTransportP2pSetup` → `ncclP2PPreconnectFunc` → `ncclGroupEnd`, failing with `Socket recv failed while polling for opId` / `Connection closed by remote peer` only after rank 0 aborted and closed its sockets (`ray_logs*/worker-*.out`).
- Flight recorder marks seq 881 `completed` on ranks 1–7 at 00:57:32Z, ~600 s after it started, i.e. at rank 0's abort, not because the exchange succeeded (`extra/h2_b3_fr_summary.txt`, `H2/probe-b3/fr-json/*.json.gz`).

**Reading:** a deterministic hang inside NCCL P2P connection setup for the first MoE all-to-all of the trainer, reproduced from b under clean conditions. Not an OOM (memory ~30 GiB/GPU during the hang), not CPU or GPU contention, and not the Ray `get_rollout_data` fetch (the data fetch had finished; all ranks reached the dispatch). The NCCL settings in effect (`NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=1`) come from the miles Ray runtime env inside the image, not from the schedule. No other settings were tried (README §H2).

**Files:** `H2/probe-b3/` holds `train.log`, `h100_config.txt`, `h2_b3_verdict.txt`, `gpu_check_node*.txt`, `nvidia-smi-start-node*.txt`, `gpu_mem_node*.csv`, `stacks/` (py-spy, two rounds per rank), `hang_stacks_node*.log`, `fr-json/` (flight recorder per rank, converted from the pickle dumps to JSON and gzipped), `ray_logs/`, `ray_logs_node1/`. All redacted with the same table as round 1.

**Gate note:** the scheduler's b3 wrapper exited before writing the gate (a `set -e` bug, fixed), so the gate still read the round-1 `FAIL b,c,d` until it was corrected by hand to `FAIL b3`. The probe itself ran to completion; only the bookkeeping was affected.

See `extra/h2_b_nccl_info_warn.txt`, multicast tracebacks, `extra/job_steps_timeline.txt`, and GPU diag in `extra/gpu_topo_nvlink_fabric.txt` when present.

## Log size notes (round 1 H2 evidence)
- `train.log` files were copied verbatim from the run tree (not head/tail truncated).
- Files over 90 MiB are stored as `*.log.gz` in this results branch.
- sacct steps <JOBID>.36–.41 (≈5 s, 2 CPU, no GPU, 2026-11-27 18:17 CET = 17:17Z): read-only overlap checks during probe b (step .35, 16:51–17:26Z); no GPU, no recipe change. Times in `extra/job_steps_timeline.txt` are CET (sacct local time).

