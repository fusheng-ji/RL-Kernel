#!/bin/bash
export H=$SCRATCH_ROOT/sched_20261127/wt/h_vi/h100-handoff/vime-dsv4-20261126
export W=$SCRATCH_ROOT/sched_20261127/vime SCRATCH=/tmp
export HF=~/miniconda3/bin/hf PY=~/datastorage/venvs/<E>/bin/python
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hfcache TMPDIR=$W/tmp; mkdir -p $TMPDIR
unset HF_TOKEN WANDB_API_KEY
echo "== setup_sources"; bash $H/setup_sources.sh; echo "setup_exit=$?"
echo "== prep_ckpt"; bash $H/prep_ckpt.sh; echo "prep_exit=$?"
echo VIME_PREP_DONE
