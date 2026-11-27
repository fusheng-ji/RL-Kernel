#!/usr/bin/env bash
# Redact stage_miles_h2_results for miles inkling2 results branch (includes ray_logs).
set -euo pipefail
D=$(dirname "$0")
# shellcheck source=/dev/null
[ -f "$D/stage_push.local.sh" ] && source "$D/stage_push.local.sh"
S=${SCHED_ROOT:-<WORK>}
ST=$S/stage_miles_h2_redacted
rm -rf "$ST"
rsync -a "$S/stage_miles_h2_results/" "$ST/"
export LEAK_USER LEAK_JOB1 LEAK_JOB2 LEAK_GROUP
export LEAK_PATH_WORK LEAK_PATH_SCRATCH_USER LEAK_PATH_HOME LEAK_PATH_PROJECT_FS
find "$ST" -type f | while read -r f; do
  STAGING_TREE="$ST" python3 - "$f" <<'PY'
import os, re, sys
p = sys.argv[1]
s = open(p, errors="replace").read()
u = os.environ.get("LEAK_USER", "<USER>")
j1 = os.environ.get("LEAK_JOB1", "<JOB_1>")
j2 = os.environ.get("LEAK_JOB2", "<JOB_2>")
g = os.environ.get("LEAK_GROUP", "<GROUP>")
def esc_path(key, repl):
    val = os.environ.get(key, "")
    if not val or "<" in val: return
    return (re.escape(val), repl)
R = []
for key, repl in (("LEAK_PATH_WORK", "<WORK>"), ("LEAK_PATH_SCRATCH_USER", "<SCRATCH>"),
                  ("LEAK_PATH_PROJECT_FS", "<PROJECT_FS>"), ("LEAK_PATH_HOME", "<HOME>")):
    pair = esc_path(key, repl)
    if pair: R.append(pair)
R += [(rf"(?<!\d){re.escape(j1)}(?!\d)", "<JOB_1>"), (rf"(?<!\d){re.escape(j2)}(?!\d)", "<JOB_2>"),
      (r"(?<!\d)59\d{5}(?!\d)", "<JOBID>"), (r"0x[0-9a-f]{16}", "<GUID>"),
      (r"\b10(?:\.\d{1,3}){3}\b", "<IP>"), (r"<CLUSTER>\d+", "node-x"),
      (r"cpu-0\d\d", "node-c")]
for var, repl in (("LEAK_PAT_SCRATCH_A", "$SCRATCH_ROOT"), ("LEAK_PAT_SCRATCH_B", "SCRATCH_ROOT"),
                  ("LEAK_REPL_E", "<E>"), ("LEAK_REPL_V", "<V>"), ("LEAK_PAT_USER_EXTRA", "<USER>"),
                  ("LEAK_PAT_NODE", "node-x"), ("LEAK_PAT_FS", "<FS>")):
    pat = os.environ.get(var, "")
    if pat and "<" not in pat: R.append((pat, repl))
if u and "<" not in u: R.append((re.escape(u), "<USER>"))
if g and "<" not in g: R.append((re.escape(g), "<GROUP>"))
for a, b in R: s = re.sub(a, b, s)
open(p, "w").write(s)
PY
done
sed -i -e 's/node-x/node-a/g' -e 's/node-x/node-b/g' "$ST/extra/job_steps_timeline.txt" 2>/dev/null || true
: "${LEAK_GREP_I:?}"
if grep -rniE "$LEAK_GREP_I" "$ST" >/tmp/miles_h2_leak.txt 2>/dev/null; then
  head -20 /tmp/miles_h2_leak.txt; exit 1
fi
echo "miles H2 redacted tree: $ST"