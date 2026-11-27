#!/usr/bin/env bash
# After jobxP3: verify P3/P4, capture nvidia-smi, T11 (node-b), V3, V4 — one at a time on JOB_X.
D=$(dirname "$0"); source "$D/common.sh"
J=${1:-$JA}
VI=$WT/h_vi/h100-handoff/vime-dsv4-20261126
log "jobx_after_p34 waiter job=$J"
wait_idle() {
  while :; do
    busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
    [ "$busy" = 0 ] && break
    sleep 60
  done
}
wait_driver_done() {
  while pgrep -f 'u_jobx_b1_chain.sh' >/dev/null || squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -q '\.7[0-9]\|jobxP3'; do sleep 120; done
  wait_idle
}
wait_driver_done
for P in P3 P4; do
  [ -f "$OUT/pipeline/$P/rows.txt" ] || { log "STOP: missing $P/rows.txt"; exit 1; }
  grep -vE ' exit=0$' "$OUT/pipeline/$P/rows.txt" | grep -q . && { log "STOP: $P rows not all exit=0"; exit 1; }
done
log "P3/P4 complete (9+5 rows exit=0)"
wait_idle
srun --jobid="$J" -w "$(nodes_of "$J" | sed -n1p)" --overlap -N1 -n1 -c 4 --gres=none --mem=0 --time=0:20:00 \
  bash -c "source '$SCHED/scripts/common.sh'; slurm_job_env_exports $J; bash $D/pipeline_capture_nvidia_smi.sh" \
  >>"$OUT/logs/pipeline_nvidia.srun.log" 2>&1
wait_idle
NB=$(nodes_of "$J" | sed -n 2p)
srun --jobid="$J" -w "$NB" --overlap -N1 -n1 -c 16 --gres=gpu:1 --exact --mem=0 --time=2:00:00 \
  bash "$D/u_t11.sh" >>"$OUT/logs/T11.srun.log" 2>&1 || { log "STOP: T11"; exit 1; }
log "T11 started/completed on node-b"
wait_idle
exp=a517384fe
got=$(git -C "$WT/h_vi" rev-parse HEAD)
[ "$got" = "$exp" ] || { log "STOP: h_vi head $got want $exp"; exit 1; }
remote=$(git ls-remote origin refs/heads/handoff/h100-vime-dsv4-20261126 2>/dev/null | awk '{print $1}')
[ "$remote" = "$exp" ] || { log "STOP: remote vime handoff $remote"; exit 1; }
NA=$(nodes_of "$J" | sed -n 1p)
driver_2node_sync "$J" "$NA" "5:00:00" vimeV3 \
  "export SLURM_JOB_ID=$J; grep -q none $GATES/vime_dsa 2>/dev/null && export DSA_BACKEND=none; bash $D/u_vime.sh V3" \
  || { log "STOP: vime V3"; exit 1; }
wait_idle
driver_2node_sync "$J" "$NA" "3:00:00" vimeV4 \
  "export SLURM_JOB_ID=$J; grep -q none $GATES/vime_dsa 2>/dev/null && export DSA_BACKEND=none; bash $D/u_vime.sh V4" \
  || { log "STOP: vime V4"; exit 1; }
log "jobx_after_p34: V3/V4 done"