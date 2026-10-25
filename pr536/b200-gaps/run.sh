#!/bin/bash
# PR #536 on four B200: the six PR test commands plus the full tests/ops suite (tuning off/on) at the PR head and at its merge base,
# the multi-GPU cases (4 GPUs visible so TP=4 runs), and the performance benchmark. Needs GAPS, WT, BASE_WT, OUT.
source "$(dirname "$0")/../../b200-gaps-common/common.sh"
env_header
export RL_KERNEL_MODE=auto
export RL_KERNEL_AUTOTUNE_CACHE_DIR=$OUT/autotune-cache
mkdir -p "$RL_KERNEL_AUTOTUNE_CACHE_DIR"; chmod 700 "$RL_KERNEL_AUTOTUNE_CACHE_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3
RL_KERNEL_AUTOTUNE=0 run gemm-off $PY -m pytest -p no:cacheprovider -q -rs tests/ops/gemm/test_det_gemm.py
RL_KERNEL_AUTOTUNE=1 run gemm-on $PY -m pytest -p no:cacheprovider -q -rs tests/ops/gemm/test_det_gemm.py
run hal-registry $PY -m pytest -p no:cacheprovider -q -rs tests/platforms/test_persistent_autotuner.py tests/runtime/test_kernel_registry.py
run operator-cache-graph $PY -m pytest -p no:cacheprovider -q -rs tests/ops/test_autotuning_gpu.py
run candidates $PY -m pytest -p no:cacheprovider -q -rs tests/ops/test_autotuning_candidates.py
run benchmark-functional $PY -m pytest -p no:cacheprovider -q -rs tests/benchmarks/test_autotuning_benchmark.py
RL_KERNEL_AUTOTUNE=0 run ops-head-off $PY -m pytest -p no:cacheprovider -q -rs tests/ops
RL_KERNEL_AUTOTUNE=1 run ops-head-on $PY -m pytest -p no:cacheprovider -q -rs tests/ops
( cd "$BASE_WT"; export PYTHONPATH=$BASE_WT
  for t in 0 1; do
    RL_KERNEL_AUTOTUNE=$t $PY -m pytest -p no:cacheprovider -q -rs tests/ops > "$OUT/ops-base-tune$t.log" 2>&1
    printf 'ops-base-tune%s\t%s\n' $t $? >> "$OUT/exitcodes.tsv"
  done )
CUDA_VISIBLE_DEVICES=0 RL_KERNEL_AUTOTUNE=0 run performance $PY -m benchmarks.tuning.benchmark_autotuning --operator all --dtype bfloat16 --preset all --phase all --trials 5 --output "$OUT/issue50-performance.json"
finish
