# P3 flight-recorder dumps: none obtained

Checked 2026-11-28T03:39Z with a `--gres=none` overlap step on both nodes while JOB_X was still running:
`ls /tmp/nccl_trace_rank_*` → "No such file or directory" on node-a and node-b (`ls-node-a.txt`, `ls-node-b.txt`);
no `*nccl_trace*` under /tmp, /dev/shm or /var/tmp either. /tmp is the node's local disk (no Slurm job_container / tmpfs),
so nothing was cleaned between runs: the dumps were never written.

The logs (pp4-tp2-s925 and pp8-s925 in P3, pp4-tp2-s925 in P3-r2) show the watchdog dump signal
("Received a dump signal due to a collective timeout ...", "Observed flight recorder dump signal from another rank via TCPStore"),
but no run set `TORCH_NCCL_TRACE_BUFFER_SIZE` (P3-r2 set only `TORCH_NCCL_DUMP_ON_TIMEOUT=1`). With the trace buffer at its
default the flight recorder most likely recorded nothing, so no `nccl_trace_rank_<N>` file was produced. The miles runs, whose
dumps exist, set `TORCH_NCCL_TRACE_BUFFER_SIZE=2000` in their own node_entry.sh.
