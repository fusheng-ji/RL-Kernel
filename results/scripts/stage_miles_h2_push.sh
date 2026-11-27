#!/usr/bin/env bash
# Pack redacted miles H2 tree for results branch (includes ray_logs); uses redact_tree.py only.
set -euo pipefail
D=$(dirname "$0")
# shellcheck source=/dev/null
[ -f "$D/stage_push.local.sh" ] && source "$D/stage_push.local.sh"
S=${SCHED_ROOT:-<SCRATCH>/sched_20261127}
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
sed -i -e 's/<NODE_A_HOST>/node-a/g' -e 's/<NODE_B_HOST>/node-b/g' \
  -e 's/<CLUSTER_NODE_PREFIX>\[[^]]*\]/node-a+node-b/g' "$ST/extra/job_steps_timeline.txt" 2>/dev/null || true
bash "$D/gzip_large_logs.sh" "$ST"
python3 - "$ST/SUMMARY.md" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
extra = """

## Log size notes (round 1 H2 evidence)
- `train.log` files were copied verbatim from the run tree (not head/tail truncated).
- Files over 90 MiB are stored as `*.log.gz` in this results branch.
- sacct steps <JOBID>.36–.41 (≈5 s, 2 CPU, no GPU, 2026-11-27T18:17Z): read-only overlap checks during probe b; no GPU, no recipe change.
"""
if p.exists():
    t = p.read_text()
    if "Log size notes" not in t:
        p.write_text(t.rstrip() + extra + "\n")
PY
bash "$D/stage_push_validate.sh" "$ST"
echo "miles H2 redacted tree: $ST"