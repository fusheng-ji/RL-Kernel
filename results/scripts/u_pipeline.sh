#!/usr/bin/env bash
# pipeline P1 P2 (1 node x 4) and P3 P4 (2 nodes x 4). usage: u_pipeline.sh P1|P2|P3|P4   (needs gate build_WP)
# P3/P4 run in a driver on node-a; srun spans both nodes (--overlap). Rows that do not fit: size 40 and record the OOM (see PLAN).
source "$(dirname "$0")/common.sh"; P=$1; O=$OUT/pipeline/$P; mkdir -p $O
gate_wait build_WP 3600 && gate_ok build_WP || exit 1
cd $WP; tree_status $WP $O/status-before.txt || exit 1
export PYTHONPATH=$WP RL_KERNEL_H3_STRICT=1 RL_KERNEL_H3_WEIGHTS=$WEIGHTS HF_HUB_OFFLINE=1 TMPDIR=${TMPDIR:-/tmp}
PY="$E1/bin/python"; NN=${SLURM_JOB_NUM_NODES:-2}
MA=$(nodes_of ${SLURM_JOB_ID:?} | sed -n 1p); export MASTER_ADDR=$MA; RUNN=0   # rank 0 is pinned to node-a with -w
# The P3/P4 driver step has no GPU (nvidia-smi: "No devices were found"); query node-a through a step that sees them.
gpu_q() { nvidia-smi "$@" 2>/dev/null | grep -q . && { nvidia-smi "$@"; return; }
  srun --jobid=${SLURM_JOB_ID:?} -w $MA --overlap -N1 -n1 -c1 --mem=0 nvidia-smi "$@" 2>&1; }
gpu_q > $O/nvidia-smi.txt; { echo "driver=$(gpu_q --query-gpu=driver_version --format=csv,noheader | head -1)"; $E1/bin/python -c "import torch;print('torch',torch.__version__)"; } > $O/env.txt
# PIPELINE_ROW_TIMEOUT (seconds, optional): cap each row so one hung row cannot use up the driver's time; exit=124 then.
run() { local n=$1 out=$2; shift 2; RUNN=$((RUNN+1)); export MASTER_PORT=$((29600+RUNN)); ${PIPELINE_ROW_TIMEOUT:+timeout -k 60 $PIPELINE_ROW_TIMEOUT} srun --jobid=${SLURM_JOB_ID:?} -w $MA --overlap -N$([ $n -gt 4 ] && echo 2 || echo 1) --ntasks=$n -c $([ $n -gt 4 ] && echo 8 || echo 16) --ntasks-per-node=$([ $n -gt 4 ] && echo 4 || echo $n) --gres=gpu:$([ $n -gt 4 ] && echo 4 || echo $n) --exact --mem=0 \
  $PY "$@" --out $O/$out > $O/$out.log 2>&1; echo "$out exit=$?" >> $O/rows.txt; }
FM="tools/validation/models/h3_parallel_full_model.py --checkpoint $CKPT --sharded"
case $P in
 P1) CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3/test_h3_pipeline.py tests/models/minimax_h3/test_h3_parallel_model.py tests/models/minimax_h3/test_h3_packed_forward.py tests/models/minimax_h3/test_h3_sharded_load.py tests/models/minimax_h3/test_h3_grid_actor.py > $O/pytest.txt 2>&1; echo "exit=$?" >> $O/pytest.txt;;
 P2) run 2 pp2-tp1-s40 $FM --pp 2 --tp 1 --size 40 --train-time-embedder
     run 4 pp4-s925 $FM --pp 4 --tp 1 --size 925 --train-time-embedder
     run 4 pp2-tp2-s925 $FM --pp 2 --tp 2 --size 925 --train-time-embedder
     run 4 pp2-tp2-sp-s928 $FM --pp 2 --tp 2 --sp --size 928
     run 4 pp2-cp2-ring-s925 $FM --pp 2 --tp 1 --cp 2 --cp-mode ring --size 925
     run 4 neg-inject $FM --pp 4 --tp 1 --size 40 --inject 30:5:2
     run 4 neg-handoff $FM --pp 4 --tp 1 --size 40 --inject-handoff 26:5
     run 4 neg-handoff-grad $FM --pp 4 --tp 1 --size 40 --inject-handoff 26:5:grad
     run 4 neg-handoff-wire $FM --pp 4 --tp 1 --size 40 --inject-handoff 26:5:wire;;
 P3) run 8 pp2-tp4-s925 $FM --pp 2 --tp 4 --size 925 --lora-rank 64 --train-time-embedder --local-size 4
     run 8 pp4-tp2-s925 $FM --pp 4 --tp 2 --size 925 --lora-rank 64 --train-time-embedder --local-size 4
     run 8 pp2-tp2-cp2-sp-s928 $FM --pp 2 --tp 2 --cp 2 --sp --size 928 --lora-rank 64 --local-size 4
     run 8 pp8-s925 $FM --pp 8 --tp 1 --size 925 --local-size 4
     run 8 tp8-s925 $FM --pp 1 --tp 8 --size 925 --lora-rank 64 --local-size 4;;
 P4) G5="tools/validation/models/h3_grid_train_step.py --checkpoint $CKPT --provider-boundaries --refiner-lora --size 928"
     run 8 g5-pp2-tp4 $G5 --pp 2 --tp 4 --repeat
     run 8 g5-pp4-tp2 $G5 --pp 4 --tp 2 --repeat --passes 5
     run 8 g5-pp2-tp2-cp2-sp $G5 --pp 2 --tp 2 --cp 2 --sp --repeat
     run 8 g5-neg $G5 --pp 2 --tp 4 --max-norm 1e9 --inject 30.attn.to_v.A:5
     run 8 g5-recipe $G5 --pp 2 --tp 4 --repeat --size recipe;;
esac
tree_status $WP $O/status-after.txt
case $P in
 P2|P3|P4)
   [ -n "${PIPELINE_CONTINUE:-}" ] && exit 0
   [ -f "$O/rows.txt" ] && grep -vE ' exit=0$' "$O/rows.txt" | grep -q . && exit 1;;
esac
