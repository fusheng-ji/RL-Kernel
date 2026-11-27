#!/usr/bin/env bash
# Allocation C: 2-node Ray placement from areal-rlkg <BUNDLE_SCRIPTS_DIR>/ + release NOTES (L1/L2 + C1-C4).
source "$(dirname "$0")/common.sh"
D=$(dirname "$0")
B=${AREAL_BUNDLE:-$H/areal-rlkg-1121}; BUNDLE_SCRIPTS=$B/<BUNDLE_SCRIPTS_DIR>
J=${SLURM_JOB_ID:-${JOB:?}}; mapfile -t N < <(nodes_of "$J"); NA=${N[0]}; NB=${N[1]}
O=$OUT/areal; mkdir -p $O/logs $O/runs
export RLK_TMPDIR=${RLK_TMPDIR:-/tmp/rlk-$(id -u)}; mkdir -p "$RLK_TMPDIR"
# shellcheck disable=SC1091
source "$B/activate.sh"
time python -c "import areal.infra.rpc.rpc_server" > $O/import-time-driver.txt 2>&1
srun --jobid=$J --overlap -w "$NB" -N1 -n1 --ntasks=1 \
  bash -c "source $B/activate.sh; /usr/bin/time -p python -c 'import areal.infra.rpc.rpc_server'" \
  > $O/import-time-node-b.txt 2>&1 || true

RAY_STEPS=$O/ray_step_ids.txt; : > "$RAY_STEPS"
stop_ray_steps() {
  while read -r sid; do [ -n "$sid" ] && scancel "$sid" 2>/dev/null || true; done < "$RAY_STEPS"
  sacct -j "$J" -S today --format=JobID,Command -P -n 2>/dev/null | grep start_ray.sh | cut -d'|' -f1 | while read -r s; do scancel "$s" 2>/dev/null; done
  : > "$RAY_STEPS"
  sleep 15
}

# Optional extra srun args per role: head_srun_extra worker_srun_extra (e.g. --cpu-bind=none)
start_ray_cluster() {
  local head_res="${1:-}" worker_res="${2:-}" tag="${3:-}"
  local head_extra="${4:-}" worker_extra="${5:-}"
  stop_ray_steps
  local head_log=$O/logs/ray_head${tag}.log worker_log=$O/logs/ray_worker${tag}.log
  # shellcheck disable=SC2086
  setsid nohup srun --jobid=$J --overlap -w "$NA" -N1 -n1 --ntasks=1 $head_extra \
    /usr/bin/env RAY_NODE_RESOURCES="$head_res" RAY_BLOCK=1 bash "$BUNDLE_SCRIPTS/start_ray.sh" head > "$head_log" 2>&1 &
  sleep 4; sacct -j "$J" -S today -n --format=JobID,Command 2>/dev/null | grep 'start_ray.sh head' | awk '{print $1}' | tail -1 >> "$RAY_STEPS" || true
  local hip=""; for _ in $(seq 1 40); do
    hip=$(grep -oE 'RAY_ADDRESS=[0-9.]+:' "$head_log" 2>/dev/null | head -1 | sed 's/RAY_ADDRESS=//;s/:$//')
    [ -n "$hip" ] && break; sleep 5
  done
  [ -n "$hip" ] || { echo "no HEAD_IP in $head_log"; return 1; }
  echo "$hip" > "$O/ray_head_ip_${tag}.txt"
  # shellcheck disable=SC2086
  setsid nohup srun --jobid=$J --overlap -w "$NB" -N1 -n1 --ntasks=1 $worker_extra \
    /usr/bin/env RAY_NODE_RESOURCES="$worker_res" RAY_BLOCK=1 bash "$BUNDLE_SCRIPTS/start_ray.sh" worker "$hip" > "$worker_log" 2>&1 &
  sleep 4; sacct -j "$J" -S today -n --format=JobID,Command 2>/dev/null | grep 'start_ray.sh worker' | awk '{print $1}' | tail -1 >> "$RAY_STEPS" || true
  for i in $(seq 1 12); do
    if srun --jobid=$J --overlap -w "$NA" -N1 -n1 --ntasks=1 bash -c \
      "unset \$(env | cut -d= -f1 | grep ^SLURM_); source $B/activate.sh; RAY_ADDRESS=${hip}:26379 timeout 90 python -c 'import ray; ray.init(); print(ray.cluster_resources().get(\"GPU\")); ray.shutdown()'" 2>/dev/null | grep -qE '^8'; then
      echo "ray_ready probe=$i tag=$tag" >> "$O/ray-probe.txt"; return 0
    fi; sleep 20
  done
  return 1
}

snap() {
  local t=$1; sleep 45
  srun --jobid=$J --overlap -w "$NA" -N1 -n1 --ntasks=1 bash -c '
    for p in $(pgrep -u '"$USER"' -f "run_trainer|vllm|rpc_server|ray::" 2>/dev/null); do
      echo "pid=$p host=$(hostname -s | md5sum | cut -c1-6) comm=$(ps -o comm= -p $p 2>/dev/null) threads=$(ls /proc/$p/task 2>/dev/null | wc -l)"
      taskset -cp $p 2>/dev/null; grep Cpus_allowed_list /proc/$p/status 2>/dev/null
    done' > "$O/placement-$t.txt" 2>&1 || true
}

run_layout() {
  local tag=$1 head_res=$2 worker_res=$3; shift 3
  export RLK_AREAL_RAY_ROLE_RESOURCES=actor=rlk_actor_node
  start_ray_cluster "$head_res" "$worker_res" "_$tag" || return 1
  local hip; hip=$(cat "$O/ray_head_ip_${tag}.txt")
  local rdir=$O/runs/$tag; mkdir -p "$rdir"
  ( snap "$tag" & )
  srun --jobid=$J --overlap -w "$NA" -N1 -n1 --ntasks=1 \
    bash "$D/areal_run_verify.sh" "$hip" "$rdir" --arms native --steps 4 "$@" \
    > "$rdir/driver.log" 2>&1 || true
  python3 - <<PY
import json,glob,os
r="$rdir"
try:
  rc=json.load(open(os.path.join(r,"native/run.json"))).get("returncode")
  n=sum(1 for _ in open(os.path.join(r,"native/stats.jsonl")))
  open(os.path.join(r,"trust.txt"),"w").write(f"returncode={rc} stats_lines={n}\n")
except Exception as e:
  open(os.path.join(r,"trust.txt"),"w").write(f"parse_err={e}\n")
PY
}

ACTOR_RES='{"rlk_actor_node": 1}'
run_layout L1-actor-on-driver "$ACTOR_RES" "" || log "L1 failed"
run_layout L2-actor-off-driver "" "$ACTOR_RES" || log "L2 failed"

python3 - "$O" <<'PY'
import json, os, sys
o = sys.argv[1]
def mean_actor(rdir):
    p = os.path.join(rdir, "native", "stats.jsonl")
    if not os.path.isfile(p):
        return None
    vals = []
    for line in open(p):
        try:
            d = json.loads(line)
            v = d.get("perf/actor_train_time") or d.get("actor_train_time")
            if v is not None:
                vals.append(float(v))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else None
l1 = mean_actor(os.path.join(o, "runs", "L1-actor-on-driver"))
l2 = mean_actor(os.path.join(o, "runs", "L2-actor-off-driver"))
note = []
slower = "L1"
actor_on_driver = 1
if l1 is None or l2 is None:
    note.append("missing stats; default L1 slower baseline")
elif l2 > l1:
    slower = "L2"; actor_on_driver = 0
if l1 and l2 and l1 > 0:
    rel = abs(l1 - l2) / max(l1, l2)
    if rel < 0.05:
        note.append(f"L1/L2 actor_train within {rel*100:.1f}% (<5%); still run C1-C3")
open(os.path.join(o, "L-baseline-note.txt"), "w").write(
    f"slower={slower} l1_mean={l1} l2_mean={l2} actor_on_driver={actor_on_driver}\n" + "\n".join(note) + "\n"
)
open(os.path.join(o, "L-baseline.env"), "w").write(f"ACTOR_ON_DRIVER={actor_on_driver}\nSLOWER={slower}\n")
PY
# shellcheck disable=SC1091
source "$O/L-baseline.env"
if [ "$ACTOR_ON_DRIVER" = 1 ]; then
  C_HEAD="$ACTOR_RES"; C_WORK=""
else
  C_HEAD=""; C_WORK="$ACTOR_RES"
fi

# C1: cpu-bind=none on head, worker, driver
stop_ray_steps
start_ray_cluster "$C_HEAD" "$C_WORK" "_C1" --cpu-bind=none --cpu-bind=none && \
  srun --jobid=$J --overlap -w "$NA" -N1 -n1 --cpu-bind=none --ntasks=1 \
    bash "$D/areal_run_verify.sh" "$(cat $O/ray_head_ip_C1.txt)" "$O/runs/C1-cpu-bind-none" --arms native --steps 4 \
    > "$O/runs/C1-cpu-bind-none/driver.log" 2>&1 || log "C1 failed"

# C2: taskset driver to cores not overlapping actor (from placement snap)
PLACEMENT=$(ls -t "$O"/placement-L*.txt 2>/dev/null | head -1)
stop_ray_steps
start_ray_cluster "$C_HEAD" "$C_WORK" "_C2" && \
  srun --jobid=$J --overlap -w "$NA" -N1 -n1 --ntasks=1 \
    bash -c "allowed=\$(awk '/^Cpus_allowed_list:/{print \$2}' /proc/self/status)
      others=\$(python3 $D/areal_pick_driver_cpus.py '$PLACEMENT' \"\$allowed\")
      taskset -c \"\$others\" bash $D/areal_run_verify.sh '$(cat $O/ray_head_ip_C2.txt)' $O/runs/C2-driver-cores --arms native --steps 4" \
    > "$O/runs/C2-driver-cores/driver.log" 2>&1 || log "C2 failed"

# C3: thread limit (slower baseline layout)
stop_ray_steps
start_ray_cluster "$C_HEAD" "$C_WORK" "_C3" && \
  srun --jobid=$J --overlap -w "$NA" -N1 -n1 --ntasks=1 \
    /usr/bin/env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    bash "$D/areal_run_verify.sh" "$(cat $O/ray_head_ip_C3.txt)" "$O/runs/C3-thread-limit" --arms native --steps 4 \
    > "$O/runs/C3-thread-limit/driver.log" 2>&1 || log "C3 failed"

gate_set areal_C4 "SKIP not attempted"
stop_ray_steps
gate_set areal_done "$(date -u +%FT%TZ)"