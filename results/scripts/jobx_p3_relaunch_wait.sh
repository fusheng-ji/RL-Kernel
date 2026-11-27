#!/usr/bin/env bash
# After jobxB1b step exits, relaunch P3/P4 via driver_2node (START=P3 chain stops after P4).
D=$(dirname "$0"); source "$D/common.sh"
J=${1:-$JA}
log "jobx P3 relaunch waiter job=$J"
while squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -qE '\.(55|68|69|70)\b'; do sleep 60; done
while :; do
  busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
  [ "$busy" = 0 ] && break
  sleep 120
done
NA=$(nodes_of "$J" | sed -n 1p)
driver_2node "$J" "$NA" "4:00:00" jobxP3 \
  "export START=P3 SLURM_JOB_ID=$J; bash $D/u_jobx_b1_chain.sh"
log "launched jobxP3 driver on job $J (P3/P4 only)"