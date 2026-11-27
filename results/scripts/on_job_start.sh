#!/usr/bin/env bash
# Auto-run: patch gate for A; user_release for C/B1/B2.
D=$(dirname "$0"); source "$D/common.sh"
J=$1; WHICH=$2   # first | second
log "on_job_start JOB=$J which=$WHICH"
bash "$D/prep_verify.sh" >> "$OUT/prep-at-run.txt" 2>&1 || log "prep_verify reported issues (see prep-at-run.txt)"
if [ "$WHICH" = first ]; then
  if gate_ok patch_autorun_A; then
    if ! gate_ok step0_A; then
      bash "$D/alloc_A.sh" "$J" || { log "STOP: alloc_A failed"; exit 1; }
    elif [ ! -f "$OUT/logs/nodeA.srun.log" ]; then
      bash "$D/resume_A_nodes.sh" "$J" || { log "STOP: resume_A_nodes failed"; exit 1; }
    else
      log "alloc_A already in progress (step0_A OK, node logs present)"
    fi
  else
    bash "$D/alloc_A_step0.sh" "$J" || { log "STOP: alloc_A_step0 failed"; exit 1; }
  fi
  gate_ok step0_A || { log "STOP: step0_A not OK, not starting B2 waiter"; exit 0; }
  (
    gate_wait_stop user_release_B2 172800
    gate_wait_stop nodea_done 36000
    gate_wait_stop nodeb_done 36000
    gate_wait_stop miles_H0 36000
    bash "$D/alloc_B2.sh" "$J" a
    gate_wait_stop miles_H5 86400
    gate_ok miles_H5 || { log "STOP: miles_H5 not OK"; exit 1; }
    bash "$D/alloc_B2.sh" "$J" b
  ) &
elif [ "$WHICH" = second ]; then
  (
    gate_wait_stop user_release_C 172800
    bash "$D/alloc_C.sh" "$J"
  ) &
  (
    gate_wait_stop user_release_B1 172800
    gate_wait_stop build_WP 36000
    gate_wait_stop vime_V2 36000
    gate_wait_stop areal_done 36000
    gate_ok nested_srun || { log "STOP: nested_srun not OK for B1"; exit 1; }
    bash "$D/alloc_B1.sh" "$J"
  ) &
fi