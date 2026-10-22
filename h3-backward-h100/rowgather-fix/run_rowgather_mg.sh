#!/bin/bash
H=<root>; D=$H/evidence/h100_sm90/h3_pr/followup
srun --overlap -w <node> --ntasks=4 bash -c "
export CUDA_HOME=$H/cuda13_home PATH=$H/cuda13_home/bin:\$PATH CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_ADDR=<node> MASTER_PORT=$((29900+RANDOM%90))
$H/venvs/rlk_h3_venv/bin/python $D/rowgather_multigpu.py $D/rowgather/multigpu_w4.json"
