#!/bin/bash
# PR #491, second round on eight B200: the all-gather benchmark at 8 ranks restricted to the operations this PR changes,
# because the first round's full 8-rank run failed in all_reduce (launch out of resources) on both the merge base and the head.
# Also records that all_reduce failure alone for both trees. Needs GAPS, WT, BASE_WT, OUT.
source "$(dirname "$0")/../../b200-gaps-common/common.sh"
env_header
B=benchmarks/benchmark_deterministic_collectives.py
run bench-after-w8-all-gather $TORCHRUN --standalone --nproc-per-node=8 $B --operations all_gather all_gather_many --output "$OUT/after_w8_all_gather.json"
( export PYTHONPATH=$BASE_WT; cd "$BASE_WT"
  $TORCHRUN --standalone --nproc-per-node=8 "$WT/$B" --operations all_gather all_gather_many --output "$OUT/before_w8_all_gather.json" > "$OUT/bench-before-w8-all-gather.log" 2>&1
  printf 'bench-before-w8-all-gather\t%s\n' $? >> "$OUT/exitcodes.tsv"
  $TORCHRUN --standalone --nproc-per-node=8 "$WT/$B" --operations all_reduce --output "$OUT/before_w8_all_reduce.json" > "$OUT/bench-before-w8-all-reduce.log" 2>&1
  printf 'bench-before-w8-all-reduce\t%s\n' $? >> "$OUT/exitcodes.tsv" )
run bench-after-w8-all-reduce $TORCHRUN --standalone --nproc-per-node=8 $B --operations all_reduce --output "$OUT/after_w8_all_reduce.json"
finish
