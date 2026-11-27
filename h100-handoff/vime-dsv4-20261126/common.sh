# Sourced by core.sbatch and drift.sbatch. Shared nodes: never pkill / ray stop; Slurm ends each srun step.
set -uo pipefail
: "${H:?export H=<THIS_DIR> before sbatch}"; source $H/env.sh
S=$W/runs/$SUITE; mkdir -p $S; R=$SCRATCH/vime-dsv4-$SLURM_JOB_ID
export DSV4_NUM_LAYERS=4 DSV4_COMPRESS_RATIOS='([0,0,4,128])' CKPT=${CKPT:-$W/ckpt/v4-4l} VIMEDIR=$W/src/vime-fix-run
# nodes sorted by IP; the first is the Ray head (its GPUs come first in vime's placement)
mapfile -t NODES < <(for n in $(scontrol show hostnames $SLURM_JOB_NODELIST); do echo "$(getent hosts $n | awk '{print $1; exit}') $n"; done | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n)
HEAD=$(echo ${NODES[0]} | awk '{print $1}')
{ date -u; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv; } > $S/gpus-$SLURM_JOB_ID.txt 2>&1
{ git -C $VIMEDIR rev-parse HEAD; git -C $VIMEDIR diff | sha256sum; git -C $W/src/Megatron-run rev-parse HEAD; git -C $W/src/Megatron-run diff | sha256sum; sha256sum $H/patches/*.patch $H/probe/v4probe.py; } > $S/pins-$SLURM_JOB_ID.txt
srun -N$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir -p $R/{data,run,tmp}; export ENROOT_MOUNT_HOME=n ENROOT_DATA_PATH=$R/data ENROOT_RUNTIME_PATH=$R/run ENROOT_TEMP_PATH=$R/tmp TMPDIR=$R/tmp; enroot create --name v4 $ENV_SQSH" > $S/create-$SLURM_JOB_ID.log 2>&1
trap "srun -N$SLURM_NNODES --ntasks-per-node=1 rm -rf $R" EXIT
VARS="CUDA_VISIBLE_DEVICES W H RUN TOPO STEPS LR EXTRA EXTRA_ENV OPT_ARGS NNODES TOPO_ARGS CKPT VIMEDIR DSV4_NUM_LAYERS DSV4_COMPRESS_RATIOS V4REWARD V4_ROUND_SYNC V4_QAT"
EARGS=""; for v in $VARS; do EARGS="$EARGS -e $v"; done
M1="--use-stateless-adam --use-precision-aware-optimizer --no-save-optim --reset-optimizer-states"
M2="--use-stateless-adam --no-save-optim --reset-optimizer-states"
DET="NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8"
B="--dsa-kernel-backend ${DSA_BACKEND:-cudnn}"

# run NAME NNODES "GPUS_NODE0 [GPUS_NODE1]" TOPO_ARGS STEPS LR EXTRA MAXMIN
# Uses exported EXTRA_ENV, OPT_ARGS, CKPT, V4REWARD, V4_ROUND_SYNC, V4_QAT. A case with an rc file is
# skipped, so a requeued job continues; rc=timeout only when the case ran out of time (exit 124).
run() {
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $1 "* ]] && return   # ONLY="T0-50 det-R0a": run just these cases
  [ -f $S/$1/rc ] && { echo "skip $1 (rc $(cat $S/$1/rc))"; return; }
  export RUN=$S/$1 NNODES=$2 TOPO=$1 TOPO_ARGS="$4" STEPS=$5 LR=$6 EXTRA="$7"; local maxmin=$8
  mkdir -p $RUN; rm -f $RUN/DONE $RUN/probe-*.jsonl
  local g=($3) i pids=() st
  echo "$(date -u +%T) START $1" >> $S/progress.txt
  for ((i=0;i<NNODES;i++)); do
    local row=(${NODES[$i]}) role=worker; [ $i = 0 ] && role=head
    timeout -k 30 $((maxmin*60)) srun -N1 -n1 --overlap -w ${row[1]} bash -c "IFN=\$(ip -o -4 addr show | grep -F ' ${row[0]}/' | awk '{print \$2; exit}'); export ENROOT_MOUNT_HOME=n ENROOT_DATA_PATH=$R/data ENROOT_RUNTIME_PATH=$R/run ENROOT_TEMP_PATH=$R/tmp TMPDIR=$R/tmp; enroot start --rw --rc $H/rc.sh -m $W:$W -m $H:$H $EARGS v4 bash $H/node_run.sh $role $HEAD ${g[$i]} ${row[0]} \$IFN" > $RUN/node$i.log 2>&1 &
    pids+=($!)
  done
  wait ${pids[0]}; st=$?; touch $RUN/DONE; wait
  [ ! -f $RUN/rc ] && [ $st = 124 ] && echo timeout > $RUN/rc
  echo "$(date -u +%T) END $1 rc=$(cat $RUN/rc 2>/dev/null || echo none)" >> $S/progress.txt
  sleep 15
}
