#!/usr/bin/env bash
# Shared environment for every allocation script. DRAFT: not run yet.
# Jobs: the two existing 2-node x 4 H100 jobs (not --exclusive, 64 CPU and 725 GB per node).
export SCHED=$SCRATCH_ROOT/sched_20261127
export H=$HOME/datastorage
export OUT=$SCHED/out WT=$SCHED/wt HAND=$SCHED/wt   # handoff checkouts: $WT/h_{img,sc,vo,pl,mi,vi,ar}
export W10=$WT/W10 WQ=$WT/WQ WB=$WT/WB WP=$WT/WP
export E1=$H/venvs/<E> E=$H/venvs/<E> V=$H/venvs/<V> E2=$H/venvs/<E2>
export IMG=$H/models/MiniMax-H3-Image-8ced6006
export CKPT=$SCRATCH_ROOT/h3ckpt/hf_full WEIGHTS=$SCRATCH_ROOT/h3ckpt RL_KERNEL_H3_WEIGHTS=$SCRATCH_ROOT/h3ckpt
export PROVIDER=$H/h3_full_h100/diffusers_src
export T9_RUN_A=$SCRATCH_ROOT/sc_20261126/out/T9/t2i/run-a
export CUDA_HOME=$H/cuda13_home PATH=$H/extra_ninja/bin:$H/cuda13_home/bin:$PATH
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.fileMode GIT_CONFIG_VALUE_0=false
unset PYTHONPATH WANDB_API_KEY HF_TOKEN
# ~/.local/bin holds a non-executable file named "env" that shadows /usr/bin/env under srun (exit 13): drop it from PATH.
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$HOME/.local/bin" | paste -sd:); export PATH
# job ids and the node order: node-a = first node of the job that runs this allocation
_lp="${BASH_SOURCE[0]%/*}/stage_push.local.sh"
[ -f "$_lp" ] && source "$_lp"
export JA=${JA:-${LEAK_JOB1:-}} JB=${JB:-${LEAK_JOB2:-}}
GATES=$OUT/gates; mkdir -p $GATES $OUT $OUT/checks $OUT/logs $OUT/vime $OUT/miles $OUT/build $OUT/runs $OUT/pipeline $OUT/areal $OUT/verlomni
log()   { echo "[$(date -u +%FT%TZ)] $*" >> $OUT/steps.txt; }
gate_set()  { echo "$(date -u +%FT%TZ) $2" > $GATES/$1; }       # gate_set NAME note
gate_ok()   { [ -f $GATES/$1 ] && ! grep -q FAIL $GATES/$1; }     # gate_ok NAME
gate_wait() { local n=$1 t=${2:-86400}; while [ $t -gt 0 ]; do [ -f $GATES/$n ] && return 0; sleep 30; t=$((t-30)); done; return 1; }
gate_wait_stop() { local n=$1 t=${2:-86400}; gate_wait "$n" "$t" || { log "STOP: gate_wait timeout $n"; exit 1; }; }
nodes_of()  { scontrol show hostnames "$(squeue -h -j $1 -o %N)"; }   # nodes_of JOBID

# Expand Cpus_allowed_list from /proc/self/status (or argument) to a comma-separated list.
read_step_cpus() {
  local list="${1:-$(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status 2>/dev/null)}"
  python3 - "$list" <<'PY'
import sys
def expand(s):
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))
cpus = expand(sys.argv[1])
if len(cpus) < 64:
    print(f"ERROR: need 64 CPUs in step cpuset, got {len(cpus)}: {sys.argv[1]}", file=sys.stderr)
    sys.exit(1)
cpus = cpus[:64]
print(",".join(map(str, cpus)))
PY
}
# Slice 64 CPUs into four taskset specs: n0,n1,n2,n3 (default 16,8,8,32).
cpu_lane_specs() {
  local n0=${1:-16} n1=${2:-8} n2=${3:-8} n3=${4:-32}
  local all; all=$(read_step_cpus) || return 1
  python3 - "$all" "$n0" "$n1" "$n2" "$n3" <<'PY'
import sys
cpus = [int(x) for x in sys.argv[1].split(",")]
n = list(map(int, sys.argv[2:6]))
if sum(n) != len(cpus):
    sys.exit(f"slice sum {sum(n)} != {len(cpus)}")
i = 0
for k in n:
    chunk = cpus[i:i+k]
    i += k
    print(",".join(map(str, chunk)))
PY
}
# Job-level SLURM exports for nested srun from login or allocation driver (unset step-local only).
slurm_job_env_exports() {
  local J=$1
  local NLIST; NLIST=$(squeue -h -j "$J" -o %N 2>/dev/null)
  unset SLURM_STEP_ID SLURM_STEP_NODELIST SLURM_STEP_NUM_NODES SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE
  unset SLURM_NTASKS SLURM_NPROCS SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_TASKS_PER_NODE SLURM_CPUS_ON_NODE
  export SLURM_JOB_ID="$J" SLURM_JOB_NODELIST="$NLIST" SLURM_NNODES=2 SLURM_JOB_NUM_NODES=2
  export SLURM_GRES=gpu:4 SLURM_OVERLAP=1
}

# Single-card step: gpu_step JOB NODE CPUS TIME NAME script [args]
# Slurm hands the step one card; the script records it as GPU (preflight) and refuses a comma list.
gpu_step() {
  local J=$1 N=$2 C=$3 T=$4 NAME=$5; shift 5
  mkdir -p $OUT/logs
  setsid nohup srun --jobid=$J -w $N --overlap -N1 -n1 -c $C --gres=gpu:1 --exact --mem=0 --time=$T \
    bash "$@" > $OUT/logs/$NAME.srun.log 2>&1 &
  disown; log "launched $NAME on job $J node $N"
}
# Whole-node step (4 cards): node_step JOB NODE CPUS TIME NAME script [args]
node_step() {
  local J=$1 N=$2 C=$3 T=$4 NAME=$5; shift 5
  mkdir -p $OUT/logs
  setsid nohup srun --jobid=$J -w $N --overlap -N1 -n1 -c $C --gres=gpu:4 --exact --mem=0 --time=$T \
    bash "$@" > $OUT/logs/$NAME.srun.log 2>&1 &
  disown; log "launched $NAME on job $J node $N (4 cards)"
}
# Run a handoff sbatch/driver script inside the allocation instead of sbatch: SLURM_JOB_* are inherited from the
# job, SLURM_OVERLAP lets its own srun calls share the nodes. #SBATCH lines are comments there.
alloc_driver() {   # alloc_driver JOB NODELIST_FIRST NAME script [args]   (env must be exported by the caller)
  local J=$1 N=$2 NAME=$3; shift 3
  setsid nohup srun --jobid=$J -w $N --overlap -N1 -n1 -c 2 --gres=none --mem=0 \
    /usr/bin/env SLURM_OVERLAP=1 bash "$@" > $OUT/logs/$NAME.driver.log 2>&1 &
  disown; log "launched driver $NAME on job $J"
}

# preflight O: card id, idle check, nvidia-smi, env.txt (only the runtime env's versions)
preflight() {
  local O=$1 PY=${2:-$V/bin/python}
  GPU=${CUDA_VISIBLE_DEVICES:?no GPU allocated}
  case "$GPU" in *,*) echo "more than one GPU: $GPU"; return 1;; esac
  mkdir -p "$O/tmp"; export TMPDIR="$O/tmp"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv -i "$GPU" > "$O/nvidia-smi.txt"
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  echo "GPU=$GPU memory_used_at_start=$used" > "$O/env.txt"
  $PY -m pip list 2>/dev/null | grep -iE '^(torch|triton|diffusers|transformers|accelerate)' >> "$O/env.txt" || true
  [ "$used" = "0" ] || { echo "GPU $GPU memory.used=$used not idle"; return 1; }
  export GPU
}
# cfile O CHECKOUT PYTHON TAG: _C must come from the checkout
cfile() { local O=$1 C=$2 P=$3 T=$4; (cd "$C" && PYTHONPATH="$C" $P/bin/python -c "import rl_engine._C as c; print(c.__file__)") > "$O/c-file-$T.txt" 2>&1
  case "$(realpath "$(tail -1 "$O/c-file-$T.txt")" 2>/dev/null)" in $(realpath "$C")/*) echo inside >> "$O/c-file-$T.txt";; *) echo NOT_inside >> "$O/c-file-$T.txt"; return 1;; esac; }
# clean tree before/after
tree_status() { git -C "$1" status --porcelain --untracked-files=no > "$2"; [ ! -s "$2" ]; }

# Two-node driver: nested srun gate OK -> in-allocation driver; else login bash with only SLURM_JOB_ID.
driver_2node() {
  local J=$1 NA=$2 T=$3 NAME=$4 CMD=$5
  mkdir -p $OUT/logs
  if gate_ok nested_srun; then
    setsid nohup srun --jobid=$J -w "$NA" --overlap -N1 -n1 -c 4 --gres=none --mem=0 --time="$T" \
      bash -c "source '$SCHED/scripts/common.sh'; slurm_job_env_exports $J; $CMD" \
      > "$OUT/logs/$NAME.driver.log" 2>&1 &
  else
  local NLIST; NLIST=$(squeue -h -j "$J" -o %N 2>/dev/null)
  setsid nohup bash -c \
    "unset SLURM_STEP_ID SLURM_STEP_NODELIST SLURM_STEP_NUM_NODES SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE
     unset SLURM_NTASKS SLURM_NPROCS SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_TASKS_PER_NODE SLURM_CPUS_ON_NODE
     export SLURM_JOB_ID=$J SLURM_JOB_NODELIST='$NLIST' SLURM_NNODES=2 SLURM_JOB_NUM_NODES=2 SLURM_GRES=gpu:4 SLURM_OVERLAP=1
     $CMD" > "$OUT/logs/$NAME.login_driver.log" 2>&1 &
  fi
  disown; log "driver_2node $NAME (nested_ok=$(gate_ok nested_srun && echo 1 || echo 0))"
}
# Same as driver_2node but blocks until the driver step finishes (for pipeline / miles round 3).
driver_2node_sync() {
  local J=$1 NA=$2 T=$3 NAME=$4 CMD=$5
  mkdir -p $OUT/logs
  if gate_ok nested_srun; then
    srun --jobid=$J -w "$NA" --overlap -N1 -n1 -c 4 --gres=none --mem=0 --time="$T" \
      bash -c "source '$SCHED/scripts/common.sh'; slurm_job_env_exports $J; $CMD" \
      >> "$OUT/logs/$NAME.driver.log" 2>&1
  else
    local NLIST; NLIST=$(squeue -h -j "$J" -o %N 2>/dev/null)
    bash -c \
      "unset SLURM_STEP_ID SLURM_STEP_NODELIST SLURM_STEP_NUM_NODES SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE
       unset SLURM_NTASKS SLURM_NPROCS SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_TASKS_PER_NODE SLURM_CPUS_ON_NODE
       export SLURM_JOB_ID=$J SLURM_JOB_NODELIST='$NLIST' SLURM_NNODES=2 SLURM_JOB_NUM_NODES=2 SLURM_GRES=gpu:4 SLURM_OVERLAP=1
       $CMD" >> "$OUT/logs/$NAME.login_driver.log" 2>&1
  fi
}
