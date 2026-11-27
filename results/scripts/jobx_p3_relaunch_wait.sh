#!/usr/bin/env bash
# After jobxB1b step exits (failed P3/P4 on 1-node srun), relaunch chain from P3 with 2-node driver.
D=$(dirname "$0"); source "$D/common.sh"
J=${1:-<JOB_1>}
while squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -q '\.55'; do sleep 60; done
busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
[ "$busy" = 0 ] || { log "jobx P3 relaunch: other steps active"; exit 0; }
NA=$(nodes_of "$J" | sed -n 1p)
export START=P3
node_step "$J" "$NA" 64 12:00:00 jobxP3 "$D/u_jobx_b1_chain.sh"