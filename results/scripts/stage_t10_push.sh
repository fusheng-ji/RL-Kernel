#!/usr/bin/env bash
# Build redacted results/T10 tree from raw $OUT (no 59xxxx JOBID rule).
set -euo pipefail
D=$(dirname "$0")
# shellcheck source=/dev/null
[ -f "$D/stage_push.local.sh" ] && source "$D/stage_push.local.sh"
S=${SCHED_ROOT:-<WORK>}
OUT=${OUT:-$S/out}
ST=$S/stage_t10_results
rm -rf "$ST"
mkdir -p "$ST"
rsync -a "$OUT/T10/" "$ST/" --exclude tmp --exclude '**/tmp/**' \
  --exclude '*.png' --exclude '*.safetensors' --exclude '*.pt'
mkdir -p "$ST/a"
cp "$OUT/build/W10/ext.txt" "$ST/a/ext.txt" 2>/dev/null || cp "$S/out/build/W10/ext.txt" "$ST/a/"
cp "$OUT/build/W10/build.log" "$ST/a/build.log" 2>/dev/null || cp "$S/out/build/W10/build.log" "$ST/a/"
export LEAK_USER LEAK_JOB1 LEAK_JOB2 LEAK_GROUP
export LEAK_PATH_WORK LEAK_PATH_SCRATCH_USER LEAK_PATH_HOME LEAK_PATH_PROJECT_FS LEAK_PATH_PROJECT_USER
export LEAK_PAT_SCRATCH_A LEAK_PAT_SCRATCH_B LEAK_REPL_E LEAK_REPL_V LEAK_REPL_E2 LEAK_REPL_ENV
export LEAK_PAT_USER_EXTRA LEAK_PAT_BUNDLE_A LEAK_PAT_BUNDLE_B LEAK_PAT_BUNDLE_C LEAK_PAT_BUNDLE_D
export LEAK_PAT_BUNDLE_E LEAK_PAT_BUNDLE_UC LEAK_PAT_PARTITION LEAK_PAT_NODE LEAK_PAT_NODES
export LEAK_PAT_CPU_PART LEAK_PAT_CLUSTER_LOGIN LEAK_PAT_CLUSTER_HOST LEAK_PAT_FS LEAK_PAT_GROUP_FS
find "$ST" -type f -print0 | while IFS= read -r -d '' f; do python3 "$D/redact_tree.py" "$f"; done
bash "$D/stage_push_validate.sh" "$ST"
if grep -r '<JOBID>' "$ST" 2>/dev/null | head -1 | grep -q .; then echo "JOBID placeholder in tree"; exit 1; fi
echo "T10 stage ready: $ST"