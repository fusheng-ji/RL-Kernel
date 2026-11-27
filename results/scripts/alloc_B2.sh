#!/usr/bin/env bash
# B2a / B2b. usage: bash alloc_B2.sh JOB a|b
D=$(dirname "$0"); source $D/common.sh; J=${1:?job id}; P=${2:?a|b}; mapfile -t N < <(nodes_of $J); NA=${N[0]}
gate_ok nested_srun || { log "B2 blocked: nested_srun gate"; exit 1; }
gate_ok miles_H0 || { log "no B2: miles_H0"; exit 1; }
if [ "$P" = a ]; then
  CMD="for h in H2 H3 H4 H5; do bash $D/u_miles.sh \$h; done"; T=08:00:00
else
  gate_ok miles_H5 || { log "no B2b: miles_H5"; exit 1; }
  CMD="bash $D/u_miles.sh H6; bash $D/u_miles.sh H7"; T=18:00:00
fi
driver_2node "$J" "$NA" "$T" "B2$P" "$CMD"