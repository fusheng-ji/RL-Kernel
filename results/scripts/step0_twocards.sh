#!/usr/bin/env bash
# Record-only: two concurrent --gres=gpu:1 steps (index only, no pci.bus_id).
source "$(dirname "$0")/common.sh"; N=$1; J=$2; O=$OUT/checks
for i in 1 2; do
  srun --jobid=$J -w $N --overlap -N1 -n1 -c 1 --gres=gpu:1 --exact --mem=0 \
    bash -c 'echo "step'$i' CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; sleep 20' \
    > $O/twocards-record-$i.txt 2>&1 &
done; wait