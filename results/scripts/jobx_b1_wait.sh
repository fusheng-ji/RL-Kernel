#!/usr/bin/env bash
# After miles H2 settles, run P1/P2 → P3/P4 → vime V2 → V3/V4 on JOB_X node-a (full B1 on JOB_X per review).
D=$(dirname "$0"); source "$D/common.sh"; J=${1:-$JA}
log "jobx B1 waiter started job=$J"
while :; do
  ohs=0; pgrep -f "on_job_start.sh $J first" >/dev/null && ohs=1
  busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
  h2=0; [ -f "$GATES/miles_H2" ] && h2=1
  [ "$h2" = 1 ] && [ "$busy" = 0 ] && [ "$ohs" = 0 ] && break
  sleep 120
done
sleep 30
busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
[ "$busy" = 0 ] || { log "jobx B1 waiter: still busy steps"; exit 0; }
NA=$(nodes_of "$J" | sed -n 1p)
log "jobx B1: launching on $NA"
node_step "$J" "$NA" 64 16:00:00 jobxB1 "$D/u_jobx_b1_chain.sh"