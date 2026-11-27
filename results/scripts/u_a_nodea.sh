#!/usr/bin/env bash
# node-a: build WP then verl-omni F0-F2, pipeline P1-P2, miles H1, optional F3 (clean timing).
D=$(dirname "$0"); source $D/common.sh
export MAX_JOBS=8
CUDA_VISIBLE_DEVICES=0 bash $D/u_build.sh WP
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash $D/u_verlomni.sh
bash $D/u_pipeline.sh P1; bash $D/u_pipeline.sh P2
gate_ok miles_H0 && bash $D/u_miles.sh H1 || log "miles H1 skipped (gate miles_H0)"
[ "${RUN_F3:-1}" = 1 ] && gate_ok verlomni_F0 && bash $D/u_verlomni.sh F3
gate_set nodea_done "$(date -u +%FT%TZ)"