#!/usr/bin/env bash
# One P3 row pp4-tp2-s925 with NCCL debug; output under P3-$TAG/.  usage: TAG=r2|r3 u_pipeline_r2_pp4.sh
#   r2: NCCL debug + TORCH_NCCL_DUMP_ON_TIMEOUT (ran 2026-11-28 01:00Z, exit=134).
#   r3: as r2, plus NCCL_RUNTIME_CONNECT=0 (connect at communicator init, no lazy connect); flight-recorder dumps
#       go to the shared P3-r3/fr/ (TORCH_NCCL_DEBUG_INFO_TEMP_FILE) with TORCH_NCCL_TRACE_BUFFER_SIZE=2000 so a
#       dump is actually recorded (r2 had no trace buffer and wrote none).
D=$(dirname "$0"); source "$D/common.sh"
TAG=${TAG:-r2}
O=$OUT/pipeline/P3-$TAG; mkdir -p "$O"
[ -s "$O/rows.txt" ] && { log "STOP: P3-$TAG/rows.txt already has rows"; exit 1; }
cd "$WP" || exit 1
tree_status "$WP" "$O/status-before.txt" || { log "STOP: P3-$TAG WP dirty"; exit 1; }
export PYTHONPATH=$WP RL_KERNEL_H3_STRICT=1 RL_KERNEL_H3_WEIGHTS=$WEIGHTS HF_HUB_OFFLINE=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,NET TORCH_NCCL_DUMP_ON_TIMEOUT=1
if [ "$TAG" = r3 ]; then
  mkdir -p "$O/fr"
  export NCCL_RUNTIME_CONNECT=0 TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$O/fr/nccl_trace_rank_ TORCH_NCCL_TRACE_BUFFER_SIZE=2000
fi
env | grep -E '^(NCCL_|TORCH_NCCL_)' | sort > "$O/nccl-env.txt"
PY="$E1/bin/python"
MA=$(nodes_of ${SLURM_JOB_ID:?} | sed -n 1p); export MASTER_ADDR=$MA; export MASTER_PORT=$([ "$TAG" = r3 ] && echo 29623 || echo 29613)
FM="tools/validation/models/h3_parallel_full_model.py --checkpoint $CKPT --sharded"
srun --jobid=${SLURM_JOB_ID:?} -w $MA --overlap -N2 --ntasks=8 -c 8 --ntasks-per-node=4 --gres=gpu:4 --exact --mem=0 \
  $PY $FM --pp 4 --tp 2 --size 925 --lora-rank 64 --train-time-embedder --local-size 4 \
  --out "$O/pp4-tp2-s925" > "$O/pp4-tp2-s925.log" 2>&1
rc=$?
echo "pp4-tp2-s925 exit=$rc" >> "$O/rows.txt"
tree_status "$WP" "$O/status-after.txt"
find "$WP" -name 'nccl_trace*.json' -newer "$O/pp4-tp2-s925.log" 2>/dev/null | head -5 | while read -r f; do cp "$f" "$O/"; done
# default dump location is each node's local /tmp/nccl_trace_rank_<N> (no .json): collect from both nodes
for n in $(nodes_of ${SLURM_JOB_ID:?}); do
  lab=$([ "$n" = "$MA" ] && echo node-a || echo node-b); mkdir -p "$O/fr-$lab"
  srun --jobid=${SLURM_JOB_ID:?} -w $n --overlap -N1 -n1 -c1 --gres=none --mem=0 \
    bash -c "ls -l --time-style=+%FT%T%z /tmp/nccl_trace_rank_* 2>&1; for f in /tmp/nccl_trace_rank_*; do [ -f \"\$f\" ] && cp -p \"\$f\" '$O/fr-$lab/'; done" \
    > "$O/fr-$lab/ls.txt" 2>&1 || true
done
log "P3-$TAG pp4-tp2-s925 exit=$rc"
