#!/usr/bin/env bash
# After jobxP3 driver: sacct/r2 → nvidia-smi → T11 → V3 → V4 (order only; not gated on pipeline pass).
D=$(dirname "$0"); source "$D/common.sh"
J=${1:-$JA}
VI=$WT/h_vi/h100-handoff/vime-dsv4-20261126
SM=$SCHED/vime/runs/smoke
log "jobx_after_p34 waiter job=$J"
wait_idle() {
  while :; do
    busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
    [ "$busy" = 0 ] && break
    sleep 60
  done
}
wait_driver_done() {
  while pgrep -f 'u_jobx_b1_chain.sh' >/dev/null || squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -qE 'jobxP3|\.7[0-9]'; do sleep 120; done
  wait_idle
}
wait_driver_done
bash "$D/collect_p3_sacct.sh" "$J"
find "$OUT/pipeline/P3" "$WP" /tmp -maxdepth 3 -name 'flight_recorder*.json' -o -name 'nccl_trace*.json' 2>/dev/null \
  | head -10 | while read -r f; do cp "$f" "$OUT/pipeline/P3/" 2>/dev/null || cp "$f" "$OUT/not-green-attempts/" 2>/dev/null; done
log "P3/P4 driver finished (rows may include failures); continuing chain"
wait_idle
NA=$(nodes_of "$J" | sed -n 1p)
driver_2node_sync "$J" "$NA" "1:30:00" pipelineR2 \
  "export SLURM_JOB_ID=$J; bash $D/u_pipeline_r2_pp4.sh"
wait_idle
srun --jobid="$J" -w "$NA" --overlap -N1 -n1 -c 4 --gres=none --mem=0 --time=0:20:00 \
  bash -c "source '$SCHED/scripts/common.sh'; slurm_job_env_exports $J; bash $D/pipeline_capture_nvidia_smi.sh" \
  >>"$OUT/logs/pipeline_nvidia.srun.log" 2>&1
wait_idle
NB=$(nodes_of "$J" | sed -n 2p)
log "T11 starting on node-b"
srun --jobid="$J" -w "$NB" --overlap -N1 -n1 -c 16 --gres=gpu:1 --exact --mem=0 --time=2:00:00 \
  bash "$D/u_t11.sh" >>"$OUT/logs/T11.srun.log" 2>&1 || { log "STOP: T11"; exit 1; }
log "T11 completed on node-b"
wait_idle
exp=a517384fe
got=$(git -C "$WT/h_vi" rev-parse HEAD)
[ "$got" = "$exp" ] || { log "STOP: h_vi head $got want $exp"; exit 1; }
remote=$(git ls-remote origin refs/heads/handoff/h100-vime-dsv4-20261126 2>/dev/null | awk '{print $1}')
[ "$remote" = "$exp" ] || { log "STOP: remote vime handoff $remote"; exit 1; }
if [ -f "$SM/preflight.json" ]; then
  {
    echo "preflight_mtime=$(stat -c %y "$SM/preflight.json" 2>/dev/null)"
    python3 - "$SM/preflight.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("cudnn_dsa=", d.get("cudnn_dsa"))
print("fp8_grid=", {k:v.get("bitwise_equal") for k,v in d.get("fp8_grid",{}).items() if isinstance(v,dict)})
PY
    grep -q none "$GATES/vime_dsa" 2>/dev/null && echo "DSA_BACKEND=none (from gate)"
  } >"$OUT/vime/V3-preflight-snapshot.txt"
fi
log "V3 starting"
driver_2node_sync "$J" "$NA" "5:00:00" vimeV3 \
  "export SLURM_JOB_ID=$J; grep -q none $GATES/vime_dsa 2>/dev/null && export DSA_BACKEND=none; bash $D/u_vime.sh V3" \
  || { log "STOP: vime V3"; exit 1; }
log "V3 completed"
wait_idle
log "V4 starting"
driver_2node_sync "$J" "$NA" "3:00:00" vimeV4 \
  "export SLURM_JOB_ID=$J; grep -q none $GATES/vime_dsa 2>/dev/null && export DSA_BACKEND=none; bash $D/u_vime.sh V4" \
  || { log "STOP: vime V4"; exit 1; }
log "V4 completed; jobx_after_p34 done"