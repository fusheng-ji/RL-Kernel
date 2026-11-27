#!/usr/bin/env bash
# Launch nodeA/nodeB after step0_A OK when patch lands mid-run. usage: bash resume_A_nodes.sh JOBID
D=$(dirname "$0"); source $D/common.sh; J=${1:?}
gate_ok step0_A || { log "resume_A_nodes: step0_A not OK"; exit 1; }
gate_ok patch_autorun_A || { log "resume_A_nodes: no patch_autorun_A"; exit 1; }
mapfile -t N < <(nodes_of $J); NA=${N[0]}; NB=${N[1]}
if [ -f "$OUT/logs/nodeA.srun.log" ] || [ -f "$OUT/logs/nodeB.srun.log" ]; then
  log "resume_A_nodes: node steps already launched"; exit 0
fi
node_step $J $NB 64 08:00:00 nodeB $D/u_nodeb.sh
node_step $J $NA 64 08:00:00 nodeA $D/u_a_nodea.sh
log "resume_A_nodes launched nodeA and nodeB"