#!/usr/bin/env bash
# Allocation A. usage: bash alloc_A.sh JOBID
D=$(dirname "$0"); source $D/common.sh; J=${1:?job id}
mapfile -t N < <(nodes_of $J); NA=${N[0]}; NB=${N[1]}
NLIST=$(squeue -h -j $J -o %N)
log "A on job $J: node-a=$NA node-b=$NB"
srun --jobid=$J -N2 --ntasks-per-node=1 --overlap --gres=none --mem=0 \
  /usr/bin/env SLURM_JOB_NODELIST="$NLIST" bash $D/step0_checks.sh
bash $D/step0_twocards.sh $NB $J
if ! bash $D/step0_nested.sh $J $NA $NB; then
  DRIVER_MODE=login bash $D/step0_nested.sh $J $NA $NB || { log "STOP: nested_srun failed"; exit 1; }
  gate_set nested_srun "OK login fallback"
fi
bash $D/step0_mustpass.sh $J || { log "STOP: step0 must-pass failed"; exit 1; }
gate_ok nested_srun || log "WARN: nested_srun not OK; B1/B2 use login fallback"
node_step $J $NB 64 08:00:00 nodeB $D/u_nodeb.sh
node_step $J $NA 64 08:00:00 nodeA $D/u_a_nodea.sh
log "alloc_A launched nodeA and nodeB"