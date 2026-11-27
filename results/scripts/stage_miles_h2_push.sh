#!/usr/bin/env bash
# Pack redacted miles H2 tree for results branch (includes ray_logs); uses redact_tree.py only.
set -euo pipefail
D=$(dirname "$0")
# shellcheck source=/dev/null
[ -f "$D/stage_push.local.sh" ] && source "$D/stage_push.local.sh"
S=${SCHED_ROOT:-<WORK>}
ST=$S/stage_miles_h2_redacted
rm -rf "$ST"
rsync -a "$S/stage_miles_h2_results/" "$ST/"
export LEAK_USER LEAK_JOB1 LEAK_JOB2 LEAK_GROUP
export LEAK_PATH_WORK LEAK_PATH_SCRATCH_USER LEAK_PATH_HOME LEAK_PATH_PROJECT_FS LEAK_PATH_PROJECT_USER
export LEAK_PAT_SCRATCH_A LEAK_PAT_SCRATCH_B LEAK_REPL_E LEAK_REPL_V LEAK_REPL_E2 LEAK_REPL_ENV
export LEAK_PAT_USER_EXTRA LEAK_PAT_BUNDLE_A LEAK_PAT_BUNDLE_B LEAK_PAT_BUNDLE_C LEAK_PAT_BUNDLE_D
export LEAK_PAT_BUNDLE_E LEAK_PAT_BUNDLE_UC LEAK_PAT_PARTITION LEAK_PAT_NODE LEAK_PAT_NODES
export LEAK_PAT_CPU_PART LEAK_PAT_CLUSTER_LOGIN LEAK_PAT_CLUSTER_HOST LEAK_PAT_FS LEAK_PAT_GROUP_FS
find "$ST" -type f -print0 | while IFS= read -r -d '' f; do
  python3 "$D/redact_tree.py" "$f"
done
sed -i -e 's/node-x/node-a/g' -e 's/node-x/node-b/g' \
  -e 's/<CLUSTER>\[[^]]*\]/node-a+node-b/g' "$ST/extra/job_steps_timeline.txt" 2>/dev/null || true
bash "$D/stage_push_validate.sh" "$ST"
echo "miles H2 redacted tree: $ST"