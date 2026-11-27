#!/bin/bash
#SBATCH -p <CPU_PARTITION>
#SBATCH --qos=cpu
#SBATCH -N1 -n1 -c 8 #SBATCH -t 04:00:00
#SBATCH --mem=120G
#SBATCH -J sched_prep
#SBATCH -o <SCRATCH>/sched_20261127/checks/cpu_job_%j.log
S=$SCRATCH_ROOT/sched_20261127; WORK=$S/miles
echo "== host"; hostname; date
echo "== enroot"; which enroot; enroot version
echo "== df"; df -h /tmp /dev/shm $S 2>&1; echo "TMPDIR=$TMPDIR"
echo "== hub"; curl -sI -m 10 https://huggingface.co | head -1; curl -sI -m 10 https://registry-1.docker.io/v2/ | head -1
echo "== import"
export ENROOT_CACHE_PATH=$WORK/cache/enroot ENROOT_TEMP_PATH=/dev/shm/enroot-tmp TMPDIR=/dev/shm/enroot-tmp   # attempt 1 used $WORK/tmp: mksquashfs 'Cannot stat' on the shared FS
mkdir -p /dev/shm/enroot-tmp
unset HF_TOKEN WANDB_API_KEY
[ -s $WORK/images/miles-inkling2-mini.sqsh ] || enroot import -o $WORK/images/miles-inkling2-mini.sqsh docker://radixark/miles:inkling-2-mini; echo import_exit=$?
sha256sum $WORK/images/miles-inkling2-mini.sqsh > $WORK/runs/h0/sqsh.sha256; cat $WORK/runs/h0/sqsh.sha256; ls -la $WORK/images
rm -rf /dev/shm/enroot-tmp
echo CPU_JOB_DONE
