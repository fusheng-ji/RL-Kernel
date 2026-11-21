#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
#
# Staged AReaL x RL-Kernel consistency and performance verification. Each stage
# writes into $OUT_ROOT/<stage>; a failing stage stops the run and everything
# collected so far is packed into $OUT_ROOT.tar.gz for offline diagnosis.
#
#   S0  inventory               packages, GPUs, source revisions (no GPU)
#   S1  preflight               every patched AReaL / Megatron / vLLM symbol (no GPU)
#   S2  smoke, graph            RL-Kernel arm only, small, vLLM CUDA graphs on
#   S3  sync, graph             native + RL-Kernel, vLLM CUDA graphs on both arms
#   S4  async, graph            same, rollout.max_head_offpolicyness=$OFFPOLICYNESS
#   S5  sync, eager             native + RL-Kernel, vLLM enforce_eager (previous setup)
#   S6  perf report             PERF.md / perf.json over all stages (no GPU)
#
# "graph" = vLLM -O0 + cudagraph_mode=FULL_DECODE_ONLY, captured at every batch
# size 1..max_num_seqs, as in the VIME CUDA experiment of RL-Kernel PR 377.
#
# Required environment:
#   AREAL_ROOT   unmodified AReaL checkout (examples/math/gsm8k_rl.py)
#   MODEL        local Qwen3 (dense, untied) HF checkpoint directory
# Optional:
#   PYTHON       interpreter with AReaL, vLLM, Megatron and rl_engine (default: python3)
#   OUT_ROOT     output root (default: ./areal-consistency-<timestamp>)
#   STEPS        steps for S3/S4/S5 (default: 10)
#   SMOKE_STEPS  steps for S2 (default: 2)
#   OFFPOLICYNESS  rollout.max_head_offpolicyness for S4 (default: 1)
#   RUN_ARGS     extra arguments for every `verify run`, e.g. "--train-tp 4 --rollout-tp 4"
#   RLK_ENV      space-separated KEY=VALUE pairs for the RL-Kernel arm
#
# Usage: run_all.sh [--from S0..S6] [--only S0..S6]
set -uo pipefail

: "${AREAL_ROOT:?set AREAL_ROOT to the AReaL checkout}"
: "${MODEL:?set MODEL to the Qwen3 checkpoint directory}"
PYTHON="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-$PWD/areal-consistency-$(date +%Y%m%d-%H%M%S)}"
STEPS="${STEPS:-10}"
SMOKE_STEPS="${SMOKE_STEPS:-2}"
OFFPOLICYNESS="${OFFPOLICYNESS:-1}"
read -r -a EXTRA <<< "${RUN_ARGS:-}"
ENV_ARGS=()
for pair in ${RLK_ENV:-}; do ENV_ARGS+=(--env "$pair"); done

FROM=S0
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

VERIFY=("$PYTHON" -m rl_engine.integrations.orchestrators.areal.verify)
mkdir -p "$OUT_ROOT"
STATUS="$OUT_ROOT/STATUS.txt"
touch "$STATUS"
echo "--- $(date '+%F %T') run_all.sh ${FROM:+--from $FROM} ${ONLY:+--only $ONLY}" >> "$STATUS"

pack() {
    tar --exclude='*/checkpoints' --exclude='*/name_resolve' \
        -czf "$OUT_ROOT.tar.gz" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")" 2>/dev/null
    echo "packed: $OUT_ROOT.tar.gz"
}
trap pack EXIT

should_run() {
    local stage="$1"
    if [[ -n "$ONLY" ]]; then [[ "$stage" == "$ONLY" ]]; return; fi
    [[ "${stage#S}" -ge "${FROM#S}" ]]
}

stage() {
    local name="$1"; shift
    should_run "$name" || { echo "$name skipped" >> "$STATUS"; return 0; }
    echo "=== $name: $*"
    local start=$SECONDS
    "$@"
    local code=$?
    echo "$name exit=$code seconds=$((SECONDS - start))" | tee -a "$STATUS"
    if [[ $code -ne 0 && $code -ne 2 ]]; then
        echo "$name failed; stopping. Results so far are in $OUT_ROOT" >&2
        exit "$code"
    fi
}

run_verify() {
    local out="$1"; shift
    "${VERIFY[@]}" run --areal-root "$AREAL_ROOT" --model "$MODEL" --out "$out" \
        "${ENV_ARGS[@]}" "${EXTRA[@]}" "$@"
}

stage S0 "${VERIFY[@]}" inventory --out "$OUT_ROOT/S0-inventory" --areal-root "$AREAL_ROOT"
stage S1 "${VERIFY[@]}" preflight --areal-root "$AREAL_ROOT" --model "$MODEL" \
    --out "$OUT_ROOT/S1-preflight"
stage S2 run_verify "$OUT_ROOT/S2-smoke-graph" --mode sync --arms rlkernel \
    --steps "$SMOKE_STEPS" --batch-size 8 --n-samples 2 --max-new-tokens 256 \
    --vllm-execution graph
stage S3 run_verify "$OUT_ROOT/S3-sync-graph" --mode sync --arms native,rlkernel \
    --steps "$STEPS" --vllm-execution graph
stage S4 run_verify "$OUT_ROOT/S4-async-graph" --mode async --arms native,rlkernel \
    --steps "$STEPS" --max-offpolicyness "$OFFPOLICYNESS" --vllm-execution graph
stage S5 run_verify "$OUT_ROOT/S5-sync-eager" --mode sync --arms native,rlkernel \
    --steps "$STEPS" --vllm-execution eager
stage S6 "${VERIFY[@]}" perf --out "$OUT_ROOT"
echo "all requested stages finished" | tee -a "$STATUS"
