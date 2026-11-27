#!/bin/bash
#SBATCH --job-name=v4-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
set -uo pipefail
: "${H:?export H=<THIS_DIR> before sbatch}"; source $H/env.sh
R=$SCRATCH/vime-dsv4-$SLURM_JOB_ID; OUT=$W/runs/smoke; mkdir -p $R/{data,run,tmp} $OUT; trap 'rm -rf $R' EXIT
# ENROOT_MOUNT_HOME=n: the host home must not appear in the container (user-site packages, old clones)
export ENROOT_MOUNT_HOME=n ENROOT_DATA_PATH=$R/data ENROOT_RUNTIME_PATH=$R/run ENROOT_TEMP_PATH=$R/tmp TMPDIR=$R/tmp
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv > $OUT/nvidia-smi.txt
enroot create --name v4 $ENV_SQSH
enroot start --rw --rc $H/rc.sh -m $W:$W -m $H:$H -e CUDA_VISIBLE_DEVICES -e W -e H v4 bash $H/smoke_inner.sh $OUT
rc=$?; echo "V2 rc=$rc (verdict in $OUT/V2-verdict.txt)"; exit $rc
