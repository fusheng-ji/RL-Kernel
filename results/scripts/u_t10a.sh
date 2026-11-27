#!/usr/bin/env bash
# T10-a (handoff h3-image-t10 README), card GPU from the step. Steps a-d then c vs T9 run-a.
source "$(dirname "$0")/common.sh"; O=$OUT/T10/a; mkdir -p $O
gate_wait build_W10 3600 && gate_ok build_W10 || { echo "W10 not built"; exit 1; }
preflight $O || exit 1; cd $W10
export RL_KERNEL_H3_IMAGE_ROOT=$IMG RL_KERNEL_H3_IMAGE_DIR=$IMG RL_KERNEL_H3_IMAGE_VAE_DIR=$IMG/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=$CKPT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
PYT="env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$W10 $V/bin/python"; STEP=tools/validation/models/h3_image_step.py; TRAJ=tools/validation/models/h3_image_trajectory.py
$PYT -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3_image > $O/pytest-suite.txt 2>&1; echo "exit=$?" >> $O/pytest-suite.txt
$PYT $STEP capture --root $IMG --text-encoder-root $CKPT --out $O/inputs > $O/capture.log 2>&1; echo "exit=$?" >> $O/capture.log
$PYT $STEP replay --root $IMG --inputs $O/inputs --out $O/run-edit --fixtures edit_two_refs > $O/run-edit.log 2>&1; echo "exit=$?" >> $O/run-edit.log
$PYT $STEP replay --root $IMG --inputs $O/inputs --out $O/run-t2i --fixtures t2i_upscaled > $O/run-t2i.log 2>&1; echo "exit=$?" >> $O/run-t2i.log
$PYT $STEP replay --root $IMG --inputs $O/inputs --out $O/run-b --fixtures t2i_upscaled --steps 0,5 --against $O/run-t2i > $O/run-b.log 2>&1; echo "exit=$?" >> $O/run-b.log
if [ -d $T9_RUN_A ]; then
  $PYT $TRAJ --root $IMG --text-encoder-root $CKPT --out $O/t2i-vs-t9 --fixtures t2i_upscaled --strict-only --against $T9_RUN_A > $O/t2i-vs-t9.log 2>&1; echo "exit=$?" >> $O/t2i-vs-t9.log
else echo "SKIPPED: previous T9 run-a is gone" > $O/t2i-vs-t9.log; fi
tree_status $W10 $O/status-after.txt
