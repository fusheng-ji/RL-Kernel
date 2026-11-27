#!/usr/bin/env bash
# JOB_Y: after areal_done and no other steps, run H2 probe b3 only (round 3).
D=$(dirname "$0"); source "$D/common.sh"
J=${JB:-<JOB_2>}
THIS=$WT/h_mi/h100-handoff/miles-inkling2-20261126
log "miles b3 waiter started job=$J"
while :; do
  gate_ok areal_done || { sleep 120; continue; }
  busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
  [ "$busy" = 0 ] || { sleep 120; continue; }
  break
done
for n in $(nodes_of "$J"); do
  other=$(squeue -h -w "$n" -o '%i' 2>/dev/null | grep -v "^$J" | grep -c . || true)
  [ "$other" = 0 ] || { log "STOP: nodes not exclusive for miles b3"; exit 1; }
done
export WORK=$SCHED/miles THIS SLURM_OVERLAP=1 TAG=b3
NA=$(nodes_of "$J" | sed -n 1p)
node_step "$J" "$NA" 64 1:30:00 milesB3 \
  "bash $D/u_miles_b3.sh $J"