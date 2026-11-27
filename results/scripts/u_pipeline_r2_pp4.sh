#!/usr/bin/env bash
# TAG r2: one P3 row pp4-tp2-s925 with NCCL debug (new driver env); output under P3-r2/.
D=$(dirname "$0"); source "$D/common.sh"
O=$OUT/pipeline/P3-r2; mkdir -p "$O"
cd "$WP" || exit 1
export PYTHONPATH=$WP RL_KERNEL_H3_STRICT=1 RL_KERNEL_H3_WEIGHTS=$WEIGHTS HF_HUB_OFFLINE=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,NET TORCH_NCCL_DUMP_ON_TIMEOUT=1
PY="$E1/bin/python"
MA=$(nodes_of ${SLURM_JOB_ID:?} | sed -n 1p); export MASTER_ADDR=$MA; export MASTER_PORT=29613
FM="tools/validation/models/h3_parallel_full_model.py --checkpoint $CKPT --sharded"
srun --jobid=${SLURM_JOB_ID:?} -w $MA --overlap -N2 --ntasks=8 -c 8 --ntasks-per-node=4 --gres=gpu:4 --exact --mem=0 \
  $PY $FM --pp 4 --tp 2 --size 925 --lora-rank 64 --train-time-embedder --local-size 4 \
  --out "$O/pp4-tp2-s925" > "$O/pp4-tp2-s925.log" 2>&1
echo "pp4-tp2-s925 exit=$?" >> "$O/rows.txt"
find "$WP" -name 'nccl_trace*.json' -newer "$O/pp4-tp2-s925.log" 2>/dev/null | head -5 | while read -r f; do cp "$f" "$O/"; done
find /tmp -name 'flight_recorder*.json' -mmin -120 2>/dev/null | head -3 | while read -r f; do cp "$f" "$O/" 2>/dev/null; done
log "P3-r2 pp4-tp2-s925 done"