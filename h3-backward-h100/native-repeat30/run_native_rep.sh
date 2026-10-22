#!/bin/bash
H=<root>; PY=$H/venvs/rlk_h3_venv/bin/python
export CUDA_HOME=$H/cuda13_home PATH=$H/cuda13_home/bin:$PATH RL_KERNEL_H3_WEIGHTS=<weights> TRITON_F32_DEFAULT=ieee CUDA_VISIBLE_DEVICES=$1
D=$H/evidence/h100_sm90/h3_pr/followup
for op in timestep_mlp adaln_projection adaln_row_gather norm_modulate gate_residual final_adaln_out; do
  case $op in timestep_mlp) b=h3-timestep-mlp-fp32;; adaln_projection) b=h3-adaln-projection-3mod;; adaln_row_gather) b=h3-adaln-row-gather;; norm_modulate) b=h3-rmsnorm;; gate_residual) b=h3-adaln-gate-residual;; final_adaln_out) b=h3-final-adaln-out;; esac
  B=$H/h100_build/$b; cd $B
  for m in native det; do
    if [ $m = det ]; then export CUBLAS_WORKSPACE_CONFIG=:4096:8; else unset CUBLAS_WORKSPACE_CONFIG; fi
    PYTHONPATH=$B $PY $D/native_diffusers_repeat.py --build $B --op $op --mode $m --n 30 --out $D/native_rep/${op}__$m.json > $D/native_rep/${op}__$m.log 2>&1 && echo ok $op $m || echo FAIL $op $m
  done
done
echo DONE
