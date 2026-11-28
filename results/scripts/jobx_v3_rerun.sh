#!/usr/bin/env bash
# Rerun V3 as SUITE=core-r2 after jobx_v3v4_only (V4) finishes.
# First V3 (SUITE=core, 2026-11-28 01:50Z) inherited CKPT=h3ckpt/hf_full from scripts/common.sh and every case
# failed at HF config load; it stays as-is under vime/runs/core and is copied to out/vime/not-green-attempts/V3-core.
# u_vime.sh now pins CKPT=$W/ckpt/v4-4l for V3/V4.
D=$(dirname "$0"); source "$D/common.sh"
J=${1:-$JA}
log "jobx_v3_rerun job=$J"
wait_idle() {
  while :; do
    busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
    [ "$busy" = 0 ] && break
    sleep 60
  done
}
exp=a517384fe
got=$(git -C "$WT/h_vi" rev-parse HEAD)
[[ "$got" == "$exp"* ]] || { log "STOP: h_vi head $got want ${exp}*"; exit 1; }
[ -d "$SCHED/vime/runs/core-r2" ] && { log "STOP: vime/runs/core-r2 already exists"; exit 1; }
NA=$(nodes_of "$J" | sed -n 1p)
wait_idle
log "V3 (SUITE=core-r2) starting"
driver_2node_sync "$J" "$NA" "8:00:00" vimeV3r2 \
  "export SLURM_JOB_ID=$J SUITE=core-r2; grep -q none $GATES/vime_dsa 2>/dev/null && export DSA_BACKEND=none; bash $D/u_vime.sh V3" \
  || { log "STOP: vime V3 core-r2"; exit 1; }
log "V3 (SUITE=core-r2) finished; jobx_v3_rerun done"
