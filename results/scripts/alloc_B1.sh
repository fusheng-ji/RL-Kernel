#!/usr/bin/env bash
# B1 (12 h): P3 -> P4 -> (vime V3 V4 if vime_V0 OK). usage: bash alloc_B1.sh JOB
D=$(dirname "$0"); source $D/common.sh; J=${1:?job id}; mapfile -t N < <(nodes_of $J); NA=${N[0]}
gate_ok nested_srun || { log "B1 blocked: nested_srun gate"; exit 1; }
gate_wait_stop build_WP 86400
T=12:00:00; gate_ok vime_V0 || T=04:00:00
V3='gate_ok vime_V0 && gate_ok vime_V2 && bash '"$D"'/u_vime.sh V3 && bash '"$D"'/u_vime.sh V4'
CMD="bash $D/u_pipeline.sh P3; bash $D/u_pipeline.sh P4; $V3"
driver_2node "$J" "$NA" "$T" B1 "$CMD"