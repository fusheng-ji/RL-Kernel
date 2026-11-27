#!/usr/bin/env bash
# C (8 h): both nodes, all CPU; Ray + verify inside u_areal.sh. usage: bash alloc_C.sh JOB
D=$(dirname "$0"); source $D/common.sh; J=${1:?job id}
mapfile -t N < <(nodes_of $J); NA=${N[0]}; NB=${N[1]}
export JOB=$J SLURM_JOB_ID=$J
setsid nohup srun --jobid=$J -w "$NA" --overlap -N1 -n1 --cpus-per-task=64 --gres=none --mem=0 --time=08:00:00 \
  /usr/bin/env JOB=$J SLURM_JOB_ID=$J bash $D/u_areal.sh > $OUT/logs/C.srun.log 2>&1 &
disown; log "alloc_C launched on job $J ($NA $NB)"