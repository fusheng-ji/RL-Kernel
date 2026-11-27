#!/usr/bin/env bash
# After P3/P4 in gpu driver: capture nvidia-smi on both nodes into pipeline P3/P4 dirs.
D=$(dirname "$0"); source "$D/common.sh"
J=${SLURM_JOB_ID:?}
NA=$(nodes_of "$J" | sed -n 1p); NB=$(nodes_of "$J" | sed -n 2p)
for P in P3 P4; do mkdir -p "$OUT/pipeline/$P"; done
srun --jobid="$J" -w "$NA" --overlap -N1 -n1 -c 8 --gres=gpu:4 --exact --mem=0 nvidia-smi \
  >"$OUT/pipeline/P3/nvidia-smi-node-a.txt" 2>&1
srun --jobid="$J" -w "$NB" --overlap -N1 -n1 -c 8 --gres=gpu:4 --exact --mem=0 nvidia-smi \
  >"$OUT/pipeline/P3/nvidia-smi-node-b.txt" 2>&1
cp "$OUT/pipeline/P3/nvidia-smi-node-a.txt" "$OUT/pipeline/P4/nvidia-smi-node-a.txt"
cp "$OUT/pipeline/P3/nvidia-smi-node-b.txt" "$OUT/pipeline/P4/nvidia-smi-node-b.txt"
echo "post-pipeline nvidia-smi captured (both nodes)" >>"$OUT/pipeline/P3/nvidia-smi.txt"
log "pipeline nvidia-smi captured on node-a and node-b"