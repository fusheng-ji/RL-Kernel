#!/bin/bash
# PR #576 on one B200: tests and the accuracy/timing bench at the PR head. Needs GAPS, WT, OUT.
source "$(dirname "$0")/../../b200-gaps-common/common.sh"
env_header
run pytest-flash-and-553 $PY -m pytest -p no:cacheprovider -q -rs \
  tests/backends/cuda/test_h3_flash_attention.py tests/backends/cuda/test_h3_full_attention.py
run pytest-flash-only $PY -m pytest -p no:cacheprovider -q -rs -v tests/backends/cuda/test_h3_flash_attention.py
for i in 1 2 3; do
  run bench-run$i $PY tools/validation/models/h3_flash_attention_bench.py --out "$OUT/bench-run$i.json"
done
finish
