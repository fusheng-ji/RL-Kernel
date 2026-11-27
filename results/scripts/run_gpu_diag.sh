#!/usr/bin/env bash
# GPU topo/nvlink/fabric on a 2-node step (after miles probes finish).
source "$(dirname "$0")/common.sh"
J=${1:-<JOB_1>}; OUT=$OUT/checks/gpu_diag; mkdir -p "$OUT"
srun --jobid="$J" -N2 --ntasks-per-node=1 --overlap --gres=gpu:4 --mem=0 bash -c '
  echo "=== $(hostname) ==="
  nvidia-smi topo -m
  nvidia-smi nvlink -s 2>/dev/null || true
  nvidia-smi -q 2>/dev/null | grep -iA3 fabric || true
' > "$OUT/gpu_topo_nvlink_fabric.txt" 2>&1
cp "$OUT/gpu_topo_nvlink_fabric.txt" "$SCHED/stage_miles_h2_results/extra/" 2>/dev/null || true