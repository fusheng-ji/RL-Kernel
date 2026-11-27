#!/bin/bash
# Runs once per node inside the container. Node 0: Ray head, waits for 8 GPUs, runs the stage. Node 1: Ray worker.
# Both: nvidia-smi and SGLang /metrics samplers (and, for H6, the live-engine checksum sidecar) for the engines on that node.
# Stops with rc 96 before anything starts if a GPU of either node is already in use (gpu_idle_check.sh).
# args: <stage script> [args...]
set -u
STAGE=$1; shift
N=${SLURM_NODEID:?}
cd /root/miles
MYIP=$(getent ahostsv4 "$(hostname)" | awk 'NR==1{print $1}')
[ "$N" = 0 ] && MYIP=$MASTER_ADDR
bash "$THIS/scripts/gpu_idle_check.sh" "$RUN/gpu_check_node$N" > "$RUN/.gpucheck_node$N" 2>&1; GC=$?
if [ "$GC" != 0 ]; then
  [ "$N" = 0 ] && touch "$RUN/.done"
  echo "node $N: $(cat "$RUN/.gpucheck_node$N"); stage not started"; exit 96
fi
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 5 > "$RUN/gpu_mem_node$N.csv" 2>/dev/null & MON="$!"
python3 "$THIS/scripts/scrape_metrics.py" "$RUN/sglang_metrics_node$N.jsonl" > "$RUN/scrape_node$N.log" 2>&1 & MON="$MON $!"
# NCCL flight recorder: on a collective timeout every rank writes its last collectives (with Python frames) to
# $RUN/fr/. Exported before `ray start` so the trainer actors on both nodes inherit it.
mkdir -p "$RUN/fr"
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000} TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
       TORCH_FR_DUMP_TEMP_FILE=$RUN/fr/node${N}_rank_
case $(basename "$STAGE") in
  h2_*|h3_*) bash "$THIS/scripts/hang_stacks.sh" "$RUN" "$RUN/stacks" "$N" > "$RUN/hang_stacks_node$N.log" 2>&1 & MON="$MON $!" ;;
esac
case $(basename "$STAGE") in
  h6_*) python3 "$THIS/scripts/sidecar_checksum.py" "$RUN/run/train.log" "$RUN/live_node$N.json" 12 > "$RUN/sidecar_node$N.log" 2>&1 & MON="$MON $!" ;;
esac
nvidia-smi > "$RUN/nvidia-smi-start-node$N.txt" 2>&1
if [ "$N" = 0 ]; then
  rm -f "$RUN/.done"
  ray stop --force >/dev/null 2>&1
  ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus 4 --disable-usage-stats > "$RUN/ray_head.log" 2>&1
  for i in $(seq 120); do
    g=$(python3 -c "import ray; ray.init(address='auto', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null)
    [ "${g:-0}" -ge 8 ] && break
    grep -q busy "$RUN/.gpucheck_node1" 2>/dev/null && break
    sleep 5
  done
  echo "ray GPUs=${g:-0}" | tee "$RUN/ray_gpus.txt"
  if grep -q busy "$RUN/.gpucheck_node1" 2>/dev/null; then
    echo "node 1: $(cat "$RUN/.gpucheck_node1"); stage not started"; rc=96
  elif [ "${g:-0}" -ge 8 ]; then
    export MILES_SCRIPT_EXTERNAL_RAY=1
    bash "$STAGE" "$@"; rc=$?
  else
    echo "worker node did not join"; rc=97
  fi
  touch "$RUN/.done"
  mkdir -p "$RUN/ray_logs" && cp -r /tmp/ray/session_latest/logs/. "$RUN/ray_logs/" 2>/dev/null
else
  for i in $(seq 120); do
    ray start --address="$MASTER_ADDR:6379" --num-gpus 4 --node-ip-address "$MYIP" --disable-usage-stats > "$RUN/ray_worker.log" 2>&1 && break
    sleep 5
  done
  while [ ! -e "$RUN/.done" ]; do sleep 10; done
  mkdir -p "$RUN/ray_logs_node1" && cp -r /tmp/ray/session_latest/logs/. "$RUN/ray_logs_node1/" 2>/dev/null
  rc=0
fi
sleep 5; kill $MON 2>/dev/null
ray stop --force >/dev/null 2>&1
exit $rc
