#!/bin/bash
#SBATCH --job-name=v4-core
#SBATCH -N2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
# V3: fix branch, M1 (the PR's optimizer), lr 1e-6. Finished cases are skipped on resubmission.
export SUITE=${SUITE:-core}; source $H/common.sh
export OPT_ARGS="$M1" EXTRA_ENV="" V4REWARD=""
# T0 / det-R0*: one node, PP1 x EP2. B200 used PP2 x EP1 (Et*3|t*1L), whose stage 0 peaked at 113 GiB (H100: 94 GB);
# a split with fewer layers on stage 0 is not allowed (all 3 hash-MoE layers must share the embedding stage).
T0A="--pipeline-model-parallel-size 1 --expert-model-parallel-size 2 --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1"
T1A="--pipeline-model-parallel-size 2 --expert-model-parallel-size 2 --pipeline-model-parallel-layout Et*3|t*1L --actor-num-nodes 2 --actor-num-gpus-per-node 2 --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1"
T2A="--pipeline-model-parallel-size 1 --expert-model-parallel-size 4 --actor-num-nodes 2 --actor-num-gpus-per-node 2 --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1"
STEPS_50=${STEPS_50:-50}
# a. 50 steps per topology; T0 also saves its rollouts for the replays below
run T0-50 1 "3"   "$T0A" $STEPS_50 1e-6 "$B --save-debug-rollout-data $S/roll/{rollout_id}.pt" 60
run T1-50 2 "2 3" "$T1A" $STEPS_50 1e-6 "$B" 60
run T2-50 2 "2 3" "$T2A" $STEPS_50 1e-6 "$B" 60
# b. deterministic replay of T0's rollout 0: one train step, weights = checkpoint, same 1024 tokens
A0="--actor-num-nodes 1 --actor-num-gpus-per-node 2 --pipeline-model-parallel-size 1 --expert-model-parallel-size 2"
A1="--actor-num-nodes 2 --actor-num-gpus-per-node 2 --pipeline-model-parallel-size 2 --expert-model-parallel-size 2 --pipeline-model-parallel-layout Et*3|t*1L"
A2="--actor-num-nodes 2 --actor-num-gpus-per-node 2 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4"
if [ -f $S/roll/0.pt ]; then
  export EXTRA_ENV="$DET"
  RP="--deterministic-mode --load-debug-rollout-data $S/roll/{rollout_id}.pt --save-debug-train-data"
  run det-R0a 1 "2"   "$A0" 1 1e-6 "$B $RP $S/train-det-R0a-{rollout_id}.pt" 25
  run det-R0b 1 "2"   "$A0" 1 1e-6 "$B $RP $S/train-det-R0b-{rollout_id}.pt" 25
  run det-R1  2 "2 2" "$A1" 1 1e-6 "$B $RP $S/train-det-R1-{rollout_id}.pt" 25
  run det-R2  2 "2 2" "$A2" 1 1e-6 "$B $RP $S/train-det-R2-{rollout_id}.pt" 25
  export EXTRA_ENV=""
else
  echo "$(date -u +%T) SKIP replays: no $S/roll/0.pt" >> $S/progress.txt
fi
# c. the PR's VIME_SKIP_GRAD_NORM at EP2 (crashed at init before the fix)
export VIME_SKIP_GRAD_NORM=1; VARS_SAVE=$EARGS; EARGS="$EARGS -e VIME_SKIP_GRAD_NORM"
run skip-T1-2 2 "2 3" "$T1A" 2 1e-6 "$B" 25
EARGS=$VARS_SAVE; unset VIME_SKIP_GRAD_NORM
cat $S/progress.txt
