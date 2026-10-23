#!/usr/bin/env bash
# Run Euler step-3 probe on B200 after: git checkout handoff/b200-euler-step3-probe && pip install -e .
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${CKPT:?export CKPT=/path/to/MiniMax-H3/hf_full}"
: "${OUT:?export OUT=/path/to/output/euler-probe-b200-seed33}"

PY="${PY:-python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT"
cd "$SCRIPT_DIR"

"$PY" h3_dual_trajectory_step3_probe.py \
  --rl-kernel-repo "$REPO_ROOT" \
  --checkpoint "$CKPT" \
  --output "$OUT" \
  --seed 33 \
  "$@"

echo "Wrote: $OUT/seed33_report.json"
