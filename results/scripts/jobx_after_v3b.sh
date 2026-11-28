#!/usr/bin/env bash
# Replaces jobx_after_v3.sh (never left its wait loop). After the V3 rerun on job J:
#   1. pipeline P4, all rows, PIPELINE_CONTINUE=1, each row capped at PIPELINE_ROW_TIMEOUT (a hung row -> exit=124,
#      the next rows still run). Code and row commands unchanged.
#   2. P3-r3 (pp4-tp2-s925 + NCCL_RUNTIME_CONNECT=0), only because r2 hung again (exit=134), and only with
#      >= R3_MIN_LEFT_MIN minutes left in the job.
#   3. AReaL alloc_C again (attempt 1 reused a stale Ray cluster; u_areal.sh fixed 2026-11-28).
# usage: bash jobx_after_v3b.sh JOB [V3_DRIVER_PID]
D=$(dirname "$0"); source "$D/common.sh"
J=${1:?job id}; WAITPID=${2:-}
log "jobx_after_v3b waiter job=$J pid=${WAITPID:-none}"
wait_idle() {
  while :; do
    busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
    [ "$busy" = 0 ] && break
    sleep 60
  done
}
left_min() { local l; l=$(squeue -h -j "$J" -o %L); python3 - "$l" <<'PY'
import sys
s = sys.argv[1]; d = 0
if "-" in s: d, s = s.split("-"); d = int(d)
p = [int(x) for x in s.split(":")]
while len(p) < 3: p.insert(0, 0)
print(d * 1440 + p[0] * 60 + p[1])
PY
}
if [ -n "$WAITPID" ]; then while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done; fi
wait_idle
mapfile -t N < <(nodes_of "$J"); NA=${N[0]}

# 1. P4
if [ -s "$OUT/pipeline/P4/rows.txt" ]; then
  log "STOP: pipeline/P4/rows.txt already has rows; move it to not-green-attempts first"; exit 1
fi
log "P4 starting (row timeout ${P4_ROW_TIMEOUT:-2700}s)"
driver_2node_sync "$J" "$NA" "5:00:00" jobxP4 \
  "export PIPELINE_CONTINUE=1 PIPELINE_ROW_TIMEOUT=${P4_ROW_TIMEOUT:-2700} SLURM_JOB_ID=$J; bash $D/u_pipeline.sh P4"
log "P4 finished: $(tr '\n' ' ' < $OUT/pipeline/P4/rows.txt 2>/dev/null)"
wait_idle

# 2. P3-r3 (optional)
r2=$(grep -o 'exit=[0-9]*' "$OUT/pipeline/P3-r2/rows.txt" 2>/dev/null | tail -1 | cut -d= -f2)
lm=$(left_min)
if [ "$r2" = 0 ]; then
  log "P3-r3 not run: r2 passed (exit=0)"
elif [ "${lm:-0}" -lt "${R3_MIN_LEFT_MIN:-600}" ]; then
  log "P3-r3 not run: ${lm} min left in job, want >= ${R3_MIN_LEFT_MIN:-600} (r3 + AReaL)"
else
  log "P3-r3 starting (r2 exit=$r2, ${lm} min left)"
  driver_2node_sync "$J" "$NA" "1:00:00" jobxP3r3 \
    "export SLURM_JOB_ID=$J TAG=r3; bash $D/u_pipeline_r2_pp4.sh"
  wait_idle
fi

# 3. AReaL: keep attempt 1 as is, then rerun alloc_C synchronously
PA=$OUT/not-green-attempts/areal_C_attempt1_stale_ray
if [ -d "$OUT/areal" ] && [ ! -e "$PA" ]; then
  mkdir -p "$(dirname "$PA")"; mv "$OUT/areal" "$PA"
  cp "$OUT/gates/areal_done" "$PA/gate-areal_done.txt" 2>/dev/null
  cp "$OUT/gates/areal_C4" "$PA/gate-areal_C4.txt" 2>/dev/null
  rm -f "$OUT/gates/areal_done" "$OUT/gates/areal_C4"
fi
log "areal alloc_C rerun starting ($(left_min) min left)"
srun --jobid=$J -w "$NA" --overlap -N1 -n1 --cpus-per-task=64 --gres=none --mem=0 --time=08:00:00 \
  /usr/bin/env JOB=$J SLURM_JOB_ID=$J bash $D/u_areal.sh > $OUT/logs/C-r2.srun.log 2>&1
log "areal alloc_C rerun exit=$? gate=$(tail -1 $OUT/gates/areal_done 2>/dev/null)"
log "jobx_after_v3b done"
