#!/usr/bin/env bash
# T10-edit: T9-edit commands at 4cfa3cbfc. Starts when <OUT>/build/W10/ext.txt begins with True. Time limit >= 6 h.
source "$(dirname "$0")/common.sh"; O=$OUT/T10/edit; mkdir -p $O
gate_wait build_W10 3600 && gate_ok build_W10 && head -c4 $OUT/build/W10/ext.txt | grep -q True || { echo "W10 ext not True"; exit 1; }
preflight $O || exit 1; cd $W10
export RL_KERNEL_H3_IMAGE_ROOT=$IMG RL_KERNEL_H3_IMAGE_DIR=$IMG RL_KERNEL_H3_IMAGE_VAE_DIR=$IMG/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=$CKPT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$W10 $V/bin/python tools/validation/models/h3_image_trajectory.py --root $IMG --text-encoder-root $CKPT --out $O/run-a \
  --fixtures edit_two_refs,edit_example --controls edit_two_refs --train-steps 5 > $O/run-a.log 2>&1; echo "exit=$?" >> $O/run-a.log
tree_status $W10 $O/status-after.txt
