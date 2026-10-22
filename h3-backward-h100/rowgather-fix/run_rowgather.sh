#!/bin/bash
H=<root>; PY=$H/venvs/rlk_h3_venv/bin/python
export CUDA_HOME=$H/cuda13_home PATH=$H/cuda13_home/bin:$PATH RL_KERNEL_H3_WEIGHTS=<weights> TRITON_F32_DEFAULT=ieee CUDA_VISIBLE_DEVICES=$1
D=$H/evidence/h100_sm90/h3_pr/followup
for op in ${2:-adaln_row_gather norm_modulate gate_residual final_adaln_out}; do
  case $op in adaln_row_gather) b=h3-adaln-row-gather;; norm_modulate) b=h3-rmsnorm;; gate_residual) b=h3-adaln-gate-residual;; final_adaln_out) b=h3-final-adaln-out;; esac
  B=$H/h100_build/$b; cd $B
  PYTHONPATH=$B $PY $D/rowgather_experiment.py --build $B --op $op --n ${3:-30} --out $D/rowgather/$op.json > $D/rowgather/$op.log 2>&1 && echo ok $op || echo FAIL $op
done
