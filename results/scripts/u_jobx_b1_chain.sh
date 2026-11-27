#!/usr/bin/env bash
# JOB_X B1 chain: P1,P2 (stop on fail) → P3,P4 → V2 idle GPU → V3,V4
D=$(dirname "$0"); source "$D/common.sh"
export CUDA_VISIBLE_DEVICES=0,1,2,3
PA=$OUT/not-green-attempts/pipeline_P1P2_jobx_attempt1; mkdir -p "$PA"
for P in P1 P2; do [ -d "$OUT/pipeline/$P" ] && mv "$OUT/pipeline/$P" "$PA/"; done
bash "$D/u_pipeline.sh" P1 || { log "STOP: P1 failed"; exit 1; }
grep -qE 'exit=0$' "$OUT/pipeline/P1/pytest.txt" || { log "STOP: P1 pytest not all pass"; exit 1; }
bash "$D/u_pipeline.sh" P2 || { log "STOP: P2 failed"; exit 1; }
grep -vE ' exit=0$' "$OUT/pipeline/P2/rows.txt" 2>/dev/null | grep -q . && { log "STOP: P2 rows not all exit=0"; exit 1; }
bash "$D/u_pipeline.sh" P3 || { log "STOP: P3 failed"; exit 1; }
bash "$D/u_pipeline.sh" P4 || { log "STOP: P4 failed"; exit 1; }
J=${SLURM_JOB_ID:?}
NA=$(scontrol show hostnames "$(squeue -h -j "$J" -o %N)" | sed -n 1p)
idle_gpu=; for g in 0 1 2 3; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  [ "$used" = "0" ] && idle_gpu=$g && break
done
[ -n "$idle_gpu" ] || { log "STOP: no idle GPU for vime V2"; exit 1; }
export SLURM_CPUS_PER_TASK=16 CPT=16
srun --jobid="$J" -w "$NA" --overlap -N1 -n1 -c 16 --gres=gpu:1 --exact --mem=0 \
  /usr/bin/env CUDA_VISIBLE_DEVICES="$idle_gpu" bash "$D/u_vime.sh" V2 || { log "STOP: vime V2"; exit 1; }
gate_ok vime_V2 || exit 1
bash "$D/u_vime.sh" V3 || { log "STOP: vime V3"; exit 1; }
bash "$D/u_vime.sh" V4 || { log "STOP: vime V4"; exit 1; }
log "jobx B1 chain done"