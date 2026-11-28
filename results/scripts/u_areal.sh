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
# Ray head/worker sruns are named areal_ray_*; cancelling those steps (own cgroup) stops only this task's Ray.
# Never `ray stop` / pkill here: other tasks (vime) run Ray on the same nodes. (sacct has no Command field on 24.05.)
ray_step_ids() { squeue -h -s -j "$J" -o '%i|%j' 2>/dev/null | awk -F'|' '$2 ~ /^areal_ray_/ {print $1}'; }
stop_ray_steps() {
  local s left
  for s in $(ray_step_ids); do echo "$s" >> "$RAY_STEPS"; scancel "$s" 2>/dev/null || true; done
  for _ in $(seq 1 12); do left=$(ray_step_ids | wc -l); [ "$left" = 0 ] && break; sleep 5; done
  [ "$left" = 0 ] || { log "areal: $left areal_ray_* steps still alive after scancel"; return 1; }
  sleep 10
}

# Optional extra srun args per role: head_srun_extra worker_srun_extra (e.g. --cpu-bind=none)
start_ray_cluster() {
  local head_res="${1:-}" worker_res="${2:-}" tag="${3:-}"
  local head_extra="${4:-}" worker_extra="${5:-}"
  stop_ray_steps || return 1
  local head_log=$O/logs/ray_head${tag}.log worker_log=$O/logs/ray_worker${tag}.log
  # shellcheck disable=SC2086
  setsid nohup srun --jobid=$J --overlap -w "$NA" -N1 -n1 --ntasks=1 -J "areal_ray_head${tag}" $head_extra \
    /usr/bin/env RAY_NODE_RESOURCES="$head_res" RAY_BLOCK=1 bash "$BUNDLE_SCRIPTS/start_ray.sh" head > "$head_log" 2>&1 &
  local hip=""; for _ in $(seq 1 40); do
    grep -q 'already running' "$head_log" 2>/dev/null && { log "areal$tag: ray head found a running cluster (see $head_log)"; return 1; }
    hip=$(grep -oE 'RAY_ADDRESS=[0-9.]+:' "$head_log" 2>/dev/null | head -1 | sed 's/RAY_ADDRESS=//;s/:$//')
    [ -n "$hip" ] && break; sleep 5
  done
  [ -n "$hip" ] || { log "areal$tag: no HEAD_IP in $head_log"; return 1; }
  echo "$hip" > "$O/ray_head_ip${tag}.txt"     # tag starts with "_": ray_head_ip_C1.txt
  # shellcheck disable=SC2086
  setsid nohup srun --jobid=$J --overlap -w "$NB" -N1 -n1 --ntasks=1 -J "areal_ray_worker${tag}" $worker_extra \
    /usr/bin/env RAY_NODE_RESOURCES="$worker_res" RAY_BLOCK=1 bash "$BUNDLE_SCRIPTS/start_ray.sh" worker "$hip" > "$worker_log" 2>&1 &
  local i gpus
  for i in $(seq 1 12); do
    grep -q 'already running' "$head_log" 2>/dev/null && { log "areal$tag: ray head found a running cluster (see $head_log)"; return 1; }
    gpus=$(srun --jobid=$J --overlap -w "$NA" -N1 -n1 --ntasks=1 bash -c \
      "unset \$(env | cut -d= -f1 | grep ^SLURM_); source $B/activate.sh; RAY_ADDRESS=${hip}:26379 timeout 90 python -c 'import ray; ray.init(); print(ray.cluster_resources().get(\"GPU\")); ray.shutdown()'" 2>/dev/null | tail -1)
    echo "probe=$i tag=$tag gpus=${gpus:-none}" >> "$O/ray-probe.txt"
    if [[ "$gpus" =~ ^8(\.0)?$ ]]; then echo "ray_ready probe=$i tag=$tag" >> "$O/ray-probe.txt"; return 0; fi
    # more than 8 GPUs: a stale node joined; waiting will not fix it
    [[ "$gpus" =~ ^[0-9.]+$ ]] && awk "BEGIN{exit !($gpus > 8)}" && { log "areal$tag: ray cluster has $gpus GPUs, want 8"; return 1; }
    sleep 20
  done
  log "areal$tag: ray cluster not ready after 12 probes (last gpus=${gpus:-none})"
  return 1
}

# trust_check RDIR: write trust.txt; ok only if native/run.json returncode==0 and stats.jsonl has rows
trust_check() {
  python3 - "$1" <<'PY'
import json, os, sys
r = sys.argv[1]
try:
    rc = json.load(open(os.path.join(r, "native/run.json"))).get("returncode")
    n = sum(1 for _ in open(os.path.join(r, "native/stats.jsonl")))
    msg = f"returncode={rc} stats_lines={n}\n"; ok = rc == 0 and n > 0
except Exception as e:
    msg = f"parse_err={e}\n"; ok = False
open(os.path.join(r, "trust.txt"), "w").write(msg)
sys.exit(0 if ok else 1)
PY
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
    > "$rdir/driver.log" 2>&1
  local st=$?
  trust_check "$rdir" || { log "areal $tag: driver exit=$st, $(cat $rdir/trust.txt)"; return 1; }
}

ACTOR_RES='{"rlk_actor_node": 1}'
run_layout L1-actor-on-driver "$ACTOR_RES" "" || log "L1 failed"
run_layout L2-actor-off-driver "" "$ACTOR_RES" || log "L2 failed"
# C1-C3 vary the slower of L1/L2; without both baselines they cannot be interpreted.
if ! { trust_check "$O/runs/L1-actor-on-driver" && trust_check "$O/runs/L2-actor-off-driver"; }; then
  stop_ray_steps
  gate_set areal_done "FAIL L1/L2 baseline missing; C1-C4 not run"
  log "STOP: areal L1/L2 baseline missing; C1-C4 not run"; exit 1
fi

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

# run_c NAME TAG DIR: start Ray, then run the driver command in "$@" (it gets HIP and RDIR in its env), check trust.
C_FAIL=""
run_c() {
  local name=$1 tag=$2 rdir=$O/runs/$3; shift 3
  mkdir -p "$rdir"
  if ! start_ray_cluster "$C_HEAD" "$C_WORK" "$tag" "${C_HEAD_EXTRA:-}" "${C_WORK_EXTRA:-}"; then
    log "$name failed: ray cluster"; C_FAIL="$C_FAIL $name"; return 1
  fi
  local hip; hip=$(cat "$O/ray_head_ip${tag}.txt")
  HIP=$hip RDIR=$rdir "$@" > "$rdir/driver.log" 2>&1
  local st=$?
  trust_check "$rdir" || { log "$name failed: driver exit=$st, $(cat $rdir/trust.txt)"; C_FAIL="$C_FAIL $name"; return 1; }
  log "$name ok: $(cat $rdir/trust.txt)"
}

# Driver commands (same as before); run_c sets HIP and RDIR.
drv_c1() { srun --jobid=$J --overlap -w "$NA" -N1 -n1 --cpu-bind=none --ntasks=1 \
  bash "$D/areal_run_verify.sh" "$HIP" "$RDIR" --arms native --steps 4; }
drv_c2() { srun --jobid=$J --overlap -w "$NA" -N1 -n1 --ntasks=1 \
  bash -c "allowed=\$(awk '/^Cpus_allowed_list:/{print \$2}' /proc/self/status)
    others=\$(python3 $D/areal_pick_driver_cpus.py '$PLACEMENT' \"\$allowed\")
    taskset -c \"\$others\" bash $D/areal_run_verify.sh '$HIP' '$RDIR' --arms native --steps 4"; }
drv_c3() { srun --jobid=$J --overlap -w "$NA" -N1 -n1 --ntasks=1 \
  /usr/bin/env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  bash "$D/areal_run_verify.sh" "$HIP" "$RDIR" --arms native --steps 4; }

# C1: cpu-bind=none on head, worker, driver
C_HEAD_EXTRA=--cpu-bind=none C_WORK_EXTRA=--cpu-bind=none run_c C1 _C1 C1-cpu-bind-none drv_c1
# C2: taskset driver to cores not overlapping actor (from placement snap)
PLACEMENT=$(ls -t "$O"/placement-L*.txt 2>/dev/null | head -1)
run_c C2 _C2 C2-driver-cores drv_c2
# C3: thread limit (slower baseline layout)
run_c C3 _C3 C3-thread-limit drv_c3

# C4 (HANDOFF step 3) moves the driver and Ray head to a CPU node; this allocation has GPU nodes only.
gate_set areal_C4 "SKIP not run: no CPU node in allocation (HANDOFF step 3)"
stop_ray_steps
if [ -n "$C_FAIL" ]; then gate_set areal_done "FAIL$C_FAIL"; else gate_set areal_done "OK C1-C3"; fi