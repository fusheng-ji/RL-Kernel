#!/usr/bin/env bash
# Allocation A step 0 only (before patch_autorun_A). usage: bash alloc_A_step0.sh JOBID
D=$(dirname "$0"); source $D/common.sh; J=${1:?job id}
mapfile -t N < <(nodes_of $J); NA=${N[0]}; NB=${N[1]}
NLIST=$(squeue -h -j $J -o %N)
log "A step0-only on job $J: node-a=$NA node-b=$NB"
srun --jobid=$J -N2 --ntasks-per-node=1 --overlap --gres=none --mem=0 \
  env SLURM_JOB_NODELIST="$NLIST" bash $D/step0_checks.sh
bash $D/step0_twocards.sh $NB $J
if ! bash $D/step0_nested.sh $J $NA $NB; then
  DRIVER_MODE=login bash $D/step0_nested.sh $J $NA $NB || { log "STOP: nested_srun failed inside and login"; exit 1; }
  gate_set nested_srun "OK login fallback"
fi
bash $D/step0_mustpass.sh $J || { log "STOP: step0 must-pass failed"; exit 1; }
log "alloc_A step0-only done (await patch_autorun_A for node lanes)"