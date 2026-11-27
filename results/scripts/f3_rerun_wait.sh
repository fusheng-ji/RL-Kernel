#!/usr/bin/env bash
# Rerun P1,P2 then verl-omni F3 (w1 fa3 + sdpa) on node-a once everything else in job X (B2a/B2b miles) has finished.
D=$(dirname "$0"); source $D/common.sh; J=${1:-<JOB_1>}
log "F3 rerun waiter started (waits for on_job_start first + no other steps in $J)"
while :; do
  busy_steps=$(squeue -h -s -j $J -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c .)
  pgrep -f "on_job_start.sh $J first" >/dev/null && ohs=1 || ohs=0
  # miles H2 FAIL means B2 stops by design (README: stop and report H2); the chain would only sit in gate_wait for miles_H5
  [ -f $GATES/miles_H2 ] && ! gate_ok miles_H2 && ohs=0
  [ "$busy_steps" = 0 ] && [ "$ohs" = 0 ] && break
  sleep 120
done
sleep 60; busy_steps=$(squeue -h -s -j $J -o '%i' | grep -v '\.batch$' | grep -c .); [ "$busy_steps" = 0 ] || exec bash $0 $J
NA=$(nodes_of $J | sed -n 1p)
log "F3 rerun: job $J idle, launching on $NA"
node_step $J $NA 64 08:00:00 f3rerun $D/u_f3_rerun.sh
