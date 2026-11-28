#!/usr/bin/env bash
# After the V3 rerun (jobx_v3_rerun.sh) on job J: pipeline P4 (never ran: the old P3 && P4 chain skipped it),
# then AReaL alloc_C again (attempt 1 reused a stale Ray cluster; u_areal.sh fixed 2026-11-28).
# usage: bash jobx_after_v3.sh JOB [V3_DRIVER_PID]
D=$(dirname "$0"); source "$D/common.sh"
J=${1:?job id}; WAITPID=${2:-}
log "jobx_after_v3 waiter job=$J pid=${WAITPID:-none}"
wait_idle() {
  while :; do
    busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
    [ "$busy" = 0 ] && break
    sleep 60
  done
}
if [ -n "$WAITPID" ]; then while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done; fi
wait_idle
mapfile -t N < <(nodes_of "$J"); NA=${N[0]}

# P4 only (P3 is final: pp4-tp2 / pp8 exit=134 reproduced 3x). PIPELINE_CONTINUE keeps rows from stopping the chain.
if [ -s "$OUT/pipeline/P4/rows.txt" ]; then
  log "STOP: pipeline/P4/rows.txt already has rows; move it to not-green-attempts first"; exit 1
fi
log "P4 starting"
driver_2node_sync "$J" "$NA" "3:00:00" jobxP4 \
  "export PIPELINE_CONTINUE=1 SLURM_JOB_ID=$J; bash $D/u_pipeline.sh P4"
log "P4 finished: $(tr '\n' ' ' < $OUT/pipeline/P4/rows.txt 2>/dev/null)"
wait_idle

# AReaL: keep attempt 1 as is, then rerun alloc_C synchronously
PA=$OUT/not-green-attempts/areal_C_attempt1_stale_ray
if [ -d "$OUT/areal" ] && [ ! -e "$PA" ]; then
  mkdir -p "$(dirname "$PA")"; mv "$OUT/areal" "$PA"
  cp "$OUT/gates/areal_done" "$PA/gate-areal_done.txt" 2>/dev/null
  cp "$OUT/gates/areal_C4" "$PA/gate-areal_C4.txt" 2>/dev/null
  rm -f "$OUT/gates/areal_done" "$OUT/gates/areal_C4"
fi
log "areal alloc_C rerun starting"
srun --jobid=$J -w "$NA" --overlap -N1 -n1 --cpus-per-task=64 --gres=none --mem=0 --time=08:00:00 \
  /usr/bin/env JOB=$J SLURM_JOB_ID=$J bash $D/u_areal.sh > $OUT/logs/C-r2.srun.log 2>&1
log "areal alloc_C rerun exit=$? gate=$(tail -1 $OUT/gates/areal_done 2>/dev/null)"
log "jobx_after_v3 done"
