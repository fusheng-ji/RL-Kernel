#!/bin/bash
# PR #491 on eight B200: the 8-GPU tests that skip on a two-GPU host, and the all-gather benchmark at the PR head.
# Extra env: BASE_WT = worktree of the merge base (same build flags) for the "before" side of the benchmark.
source "$(dirname "$0")/../../b200-gaps-common/common.sh"
env_header
T="tests/distributed/test_deterministic_all_gather_sizes.py tests/distributed/collectives/test_deterministic_all_gather.py tests/distributed/collectives/test_transport_deterministic_collective.py"
T2="tests/distributed/collectives/test_deterministic_all_reduce.py tests/distributed/collectives/test_deterministic_reduce_scatter.py"
run pytest-8gpu-all-gather $PY -m pytest -p no:cacheprovider -v -rs $T
run pytest-8gpu-reduce $PY -m pytest -p no:cacheprovider -v -rs $T2
CUDA_VISIBLE_DEVICES=0,1 run pytest-2gpu-all-gather $PY -m pytest -p no:cacheprovider -v -rs $T
for w in 2 8; do
  run bench-after-w$w $TORCHRUN --standalone --nproc-per-node=$w benchmarks/benchmark_deterministic_collectives.py --output "$OUT/after_w$w.json"
  ( export PYTHONPATH=$BASE_WT; cd "$BASE_WT"
    printf '%s\t%s\n' "bench-before-w$w" "torchrun benchmark (merge base)" >> "$OUT/commands.tsv"
    $TORCHRUN --standalone --nproc-per-node=$w "$WT/benchmarks/benchmark_deterministic_collectives.py" --output "$OUT/before_w$w.json" > "$OUT/bench-before-w$w.log" 2>&1
    printf '%s\t%s\n' "bench-before-w$w" "$?" >> "$OUT/exitcodes.tsv" )
done
finish
