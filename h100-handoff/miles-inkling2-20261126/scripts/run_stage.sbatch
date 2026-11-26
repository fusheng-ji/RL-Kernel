#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=0
# Two nodes x 4 H100, one container per node. Partition, account, time and --output are given on the sbatch line.
# usage: sbatch --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=<tag> [--time=..] run_stage.sbatch <stage script> [args...]
# --exclusive: the miles launcher runs pkill -9 sglang/miles/redis and enroot does not isolate PIDs.
set -u
: "${WORK:?}" "${THIS:?}" "${TAG:?}"
STAGE=$1; shift
export RUN=$WORK/runs/$(basename "$STAGE" .sh)-$TAG
mkdir -p "$RUN"
HEAD_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_ADDR=$(getent ahostsv4 "$HEAD_HOST" | awk 'NR==1{print $1}')
echo "nodes=$SLURM_JOB_NUM_NODES head_ip_set=$([ -n "$MASTER_ADDR" ] && echo yes || echo no)"
df -h / /tmp /dev/shm "$WORK" 2>/dev/null
srun --ntasks-per-node=1 --container-image="$WORK/images/miles-inkling2-mini.sqsh" --container-remap-root --no-container-mount-home \
     --container-writable --container-mounts="$WORK:$WORK,$THIS:$THIS" --container-workdir=/root \
     bash -c "unset WANDB_API_KEY HF_TOKEN; export HOME=/root HF_HOME=$WORK/cache/hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 \
              WORK=$WORK THIS=$THIS RUN=$RUN MASTER_ADDR=$MASTER_ADDR TMPDIR=/tmp; bash $THIS/scripts/node_entry.sh $STAGE $*"
rc=$?
echo "stage rc=$rc"
exit $rc
