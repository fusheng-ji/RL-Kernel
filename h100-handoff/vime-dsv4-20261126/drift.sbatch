#!/bin/bash
#SBATCH --job-name=v4-drift
#SBATCH -N2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
# V4: train/rollout drift with FP32 main params (M2), lr 1e-5, 24 steps; FP8 as is vs training-side QAT.
# Trainer: 4 GPUs on the head node, PP1 x EP4. B200 used PP2 x EP2 (Et*3|t*1L) on one node, whose stage 0
# peaked at 116-119 GiB (H100: 94 GB); the 3 hash-MoE layers must stay on the embedding stage, so no other split.
# Rollout: 1 GPU on the second node.
export SUITE=${SUITE:-drift}; source $H/common.sh
export OPT_ARGS="$M2" EXTRA_ENV="" V4REWARD=lower
STEPS_24=${STEPS_24:-24}
TQ="--pipeline-model-parallel-size 1 --expert-model-parallel-size 4 --actor-num-nodes 1 --actor-num-gpus-per-node 4 --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1"
case_() { # NAME CKPT ROUND_SYNC QAT
  export CKPT=$2 V4_ROUND_SYNC=$3 V4_QAT=$4
  run $1 2 "4 1" "$TQ" $STEPS_24 1e-5 "$B" 45
}
for c in ${CASES:-fp8-M2-a qat-M2-a fp8-M2-b qat-M2-b}; do
  case $c in
    fp8-*) case_ $c $W/ckpt/v4-4l "" "";;
    qat-*) case_ $c $W/ckpt/v4-4l "" 1;;
    bf16-*) case_ $c $W/ckpt/v4-4l-bf16 "" "";;   # optional, needs V1's BF16 checkpoint
    rnd-*) case_ $c $W/ckpt/v4-4l-bf16 1 "";;     # optional: BF16 rollout, synced weights rounded to the FP8 grid
  esac
done
cat $S/progress.txt
