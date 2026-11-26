#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=0
# One node x 4 H100, for stages that never start the miles launcher across nodes (H1, H4b, H7 fresh scoring).
# usage: sbatch --export=ALL,WORK=<WORK>,THIS=<THIS_DIR>,TAG=<tag> [--time=..] run_stage_1node.sbatch <stage script> [args...]
set -u
: "${WORK:?}" "${THIS:?}" "${TAG:?}"
STAGE=$1; shift
export RUN=$WORK/runs/$(basename "$STAGE" .sh)-$TAG
mkdir -p "$RUN"
df -h / /tmp /dev/shm "$WORK" 2>/dev/null
srun --container-image="$WORK/images/miles-inkling2-mini.sqsh" --container-remap-root --no-container-mount-home \
     --container-writable --container-mounts="$WORK:$WORK,$THIS:$THIS" --container-workdir=/root \
     bash -c "unset WANDB_API_KEY HF_TOKEN; export HOME=/root HF_HOME=$WORK/cache/hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 \
              WORK=$WORK THIS=$THIS RUN=$RUN TMPDIR=/tmp; nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 5 > $RUN/gpu_mem_node0.csv 2>/dev/null & \
              bash $STAGE $*; rc=\$?; kill %1 2>/dev/null; exit \$rc"
rc=$?
echo "stage rc=$rc"
exit $rc
