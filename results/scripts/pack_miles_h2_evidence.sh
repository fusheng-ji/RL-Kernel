#!/usr/bin/env bash
# Pack H2 evidence for handoff/h100-miles-inkling2-results-20261126 (run after miles_H2 gate).
set -euo pipefail
D=$(dirname "$0"); source "$D/common.sh"
WORK=$SCHED/miles
THIS=$WT/h_mi/h100-handoff/miles-inkling2-20261126
DEST=$SCHED/stage_miles_h2_results
rm -rf "$DEST"; mkdir -p "$DEST/H2" "$DEST/extra"
H2DIRS="$WORK/runs/h2_memprobe-a $WORK/runs/h2_memprobe-b $WORK/runs/h2_memprobe-c $WORK/runs/h2_memprobe-d"
python3 "$THIS/scripts/probe_summary.py" $H2DIRS > "$DEST/H2/h2_probe_summary.json"
for letter in a b c d; do
  st="$WORK/runs/h2_memprobe-$letter"
  tag=$(ls "$st" 2>/dev/null | head -1)
  [ -n "$tag" ] || continue
  sub="$st/$tag"
  mkdir -p "$DEST/H2/probe-$letter"
  cp "$sub/train.log" "$sub/h100_config.txt" "$DEST/H2/probe-$letter/"
  cp "$st"/gpu_mem_node*.csv "$DEST/H2/probe-$letter/" 2>/dev/null || true
  for rdir in "$st/ray_logs" "$st/ray_logs_node1" "$st/$tag/ray_logs"; do
    [ -d "$rdir" ] && rsync -a "$rdir" "$DEST/H2/probe-$letter/" 2>/dev/null || true
  done
done
grep -nE 'NCCL (INFO|WARN)|ProcessGroupNCCL' "$WORK/runs/h2_memprobe-b/b/train.log" \
  > "$DEST/extra/h2_b_nccl_info_warn.txt" || true
for letter in c d; do
  grep -nA20 'requires NVLink multicast' "$WORK/runs/h2_memprobe-$letter"/*/train.log 2>/dev/null \
    > "$DEST/extra/h2_${letter}_nvlink_multicast_traceback.txt" || true
done
sacct -j "${JA:-<JOB_1>}" --format=JobID,Start,End,NodeList -P \
  | sed -e 's/node-x/node-a/g' -e 's/node-x/node-b/g' \
  > "$DEST/extra/job_steps_timeline.txt"
cat > "$DEST/SUMMARY.md" <<EOF
# miles H2 H100 probe (JOB_X)

Gate: $(cat "$GATES/miles_H2" 2>/dev/null || echo pending)

- **a** (TP1 0.55): engine start failed, hybrid state cache negative; train rc=124; no perf 0.
- **b** (TP1 0.80): engines OK; one perf 0; NCCL ALLTOALL watchdog timeout at seq 881; train rc=1; peaks ~94 GiB/GPU.
- **c,d** (TP2): SGLang \`Inkling2 tensor parallelism requires NVLink multicast\`; no perf 0.

No other settings tried; no offload (README §H2). H3–H7 not run.

See \`extra/h2_b_nccl_info_warn.txt\`, multicast tracebacks, \`extra/job_steps_timeline.txt\`, and GPU diag in \`extra/gpu_topo_nvlink_fabric.txt\` when present.
EOF
echo "packed $DEST"