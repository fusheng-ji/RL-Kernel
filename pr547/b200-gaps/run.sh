#!/bin/bash
# PR #547 on one B200: the repo-reproducible commands from the PR body, run with the shared torch 2.13 environment at the PR head.
# Needs GAPS, WT, OUT.
source "$(dirname "$0")/../../b200-gaps-common/common.sh"
env_header
export CUDA_VISIBLE_DEVICES=0
run pytest-recursive $PY -m pytest -p no:cacheprovider -q -rs tests/ops/logprob/test_deterministic_logp_recursive.py
RL_KERNEL_LOGP_MAX_ORDER=10 run pytest-order10 $PY -m pytest -p no:cacheprovider -q -rs \
  tests/ops/logprob/test_deterministic_logp_recursive.py -k 'recursive_orders and dtype3'
run pytest-all-logprob $PY -m pytest -p no:cacheprovider -q -rs tests/ops/logprob
run pytest-layout $PY -m pytest -p no:cacheprovider -q -rs tests/runtime/test_layout_compatibility.py
run probe $PY tools/validation/operators/probe_deterministic_logp_autograd.py --output "$OUT/recursive-backends.json"
finish
