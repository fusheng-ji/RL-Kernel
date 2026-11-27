#!/usr/bin/env bash
# Build redacted stage tree for handoff results branches. Does not push.
# Committed copy: site paths and identity come from stage_push.local.sh (not committed).
set -euo pipefail
D=$(dirname "$0")
# shellcheck source=/dev/null
[ -f "$D/stage_push.local.sh" ] && source "$D/stage_push.local.sh"
S=${SCHED_ROOT:-<SCRATCH>/sched_20261127}
ST=$S/stage_out
LEAK_USER=${LEAK_USER:-<USER>}
LEAK_JOB1=${LEAK_JOB1:-<JOB_1>}
LEAK_JOB2=${LEAK_JOB2:-<JOB_2>}
LEAK_GROUP=${LEAK_GROUP:-<GROUP>}

rm -rf "$ST"
mkdir -p "$ST"/{scripts,checks,not-green-attempts,partial-results}
cp "$S/stage/PLAN.md" "$ST/PLAN.md"
[ -f "$S/stage/RUN_SUMMARY.md" ] && cp "$S/stage/RUN_SUMMARY.md" "$ST/RUN_SUMMARY.md"
for f in "$S/scripts"/*.sh "$S/scripts"/*.sbatch; do
  [ -f "$f" ] || continue
  case "$f" in *.local.sh) continue ;; esac
  cp "$f" "$ST/scripts/"
done
cp "$S/scripts/areal_pick_driver_cpus.py" "$ST/scripts/" 2>/dev/null || true
OUT=${OUT:-$S/out}
if [ -d "$OUT" ]; then
  mkdir -p "$ST/checks/run"
  cp -a "$OUT/checks"/* "$ST/checks/run/" 2>/dev/null || true
  { for g in "$OUT/gates"/*; do [ -f "$g" ] && echo "$(basename "$g"): $(cat "$g")"; done; } > "$ST/checks/gates-snapshot.txt"
  [ -f "$OUT/prep-at-run.txt" ] && cp "$OUT/prep-at-run.txt" "$ST/checks/prep-at-run.txt"
  [ -f "$OUT/steps.txt" ] && cp "$OUT/steps.txt" "$ST/checks/steps.txt"
  [ -f "$S/checks/progress_watch.log" ] && tail -300 "$S/checks/progress_watch.log" > "$ST/checks/progress_watch-tail.log"
  for name in miles pipeline QK2 vime build; do
    [ -d "$OUT/$name" ] && rsync -a --exclude 'tmp/' --exclude '**/tmp/**' \
      "$OUT/$name/" "$ST/partial-results/$name/" 2>/dev/null || true
  done
  find "$OUT/T10" -type f \( -name '*.log' -o -name '*.txt' -o -name '*.json' -o -name 'nvidia-smi.txt' \) \
    ! -path '*/tmp/*' 2>/dev/null \
    | while read -r f; do d="$ST/partial-results/T10/${f#$OUT/T10/}"; mkdir -p "$(dirname "$d")"; cp "$f" "$d"; done
  rsync -a "$OUT/verlomni/F0" "$OUT/verlomni/"*.json "$OUT/verlomni/"*.log "$OUT/verlomni/steps.txt" \
    "$ST/partial-results/verlomni/" 2>/dev/null || true
  find "$OUT/verlomni/runs" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' -o -name '*.txt' -o -name 'preflight.json' -o -name 'trust.txt' \) 2>/dev/null \
    | while read -r f; do d="$ST/partial-results/verlomni/runs/${f#$OUT/verlomni/runs/}"; mkdir -p "$(dirname "$d")"; cp "$f" "$d"; done
fi
if [ -d "$S/stage_miles_h2_results" ]; then
  mkdir -p "$ST/partial-results/miles-h2"
  rsync -a --exclude 'ray_logs/' --exclude 'ray_logs_node1/' --exclude 'probe-*/ray_logs/' \
    "$S/stage_miles_h2_results/" "$ST/partial-results/miles-h2/"
fi
# Full vime freeze artifacts (re-copied each push; redaction must not break version pins in == fields)
for vf in env-freeze-sm90.txt base-image-freeze.txt pip-list.txt; do
  [ -f "$S/vime/wheels-sm90/$vf" ] && cp "$S/vime/wheels-sm90/$vf" "$ST/partial-results/vime/" 2>/dev/null || true
done
[ -f "$OUT/vime/env-freeze-sm90.txt" ] && cp "$OUT/vime/env-freeze-sm90.txt" "$ST/partial-results/vime/" 2>/dev/null || true
cp "$S/not-green-attempts"/* "$ST/not-green-attempts/" 2>/dev/null || true
[ -d "$OUT/not-green-attempts" ] && rsync -a --exclude "tmp/" "$OUT/not-green-attempts/" "$ST/not-green-attempts/" 2>/dev/null || true

for f in prep-verify.txt test_only_N2_exclusive_18h.txt miles_dl.log vime_prep.log verlomni_prep.log cpu_job.log sqsh-superblock.txt monitor.log; do
  [ -f "$S/checks/$f" ] && cp "$S/checks/$f" "$ST/checks/"
done
cp "$S/areal-release/areal-ropefusion-placement-run/NOTES.md" "$ST/checks/areal-placement-NOTES.md" 2>/dev/null || true
[ -f "$S/checks/cpu_job.log" ] && cp "$S/checks/cpu_job.log" "$ST/checks/"

for f in "$ST/checks/cpu_job.log" "$ST/not-green-attempts/cpu_job.log"; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import re, sys
p = sys.argv[1]
lines = open(p).read().splitlines()
if len(lines) >= 3 and re.search(r"\d{2}:\d{2}:\d{2}", lines[2]) and "T" not in lines[2]:
    lines[2] = "[UTC] " + re.sub(r"^\S+\s+\S+\s+", "", lines[2]).strip()
open(p, "w").write("\n".join(lines) + ("\n" if lines else ""))
PY
done

for f in "$ST/not-green-attempts"/*.log; do
  [ -f "$f" ] || continue
  { grep -v 'Cannot stat' "$f" || true
    echo "[... $(grep -c 'Cannot stat' "$f" 2>/dev/null || echo 0) mksquashfs 'Cannot stat ... ignoring' lines removed ...]"
  } > "$f.tmp" && mv "$f.tmp" "$f"
done

export STAGING_TREE="$ST"
export LEAK_USER LEAK_JOB1 LEAK_JOB2 LEAK_GROUP
export LEAK_PATH_WORK LEAK_PATH_SCRATCH_USER LEAK_PATH_HOME LEAK_PATH_PROJECT_FS LEAK_PATH_PROJECT_USER
export LEAK_PAT_SCRATCH_A LEAK_PAT_SCRATCH_B LEAK_REPL_E LEAK_REPL_V LEAK_REPL_E2 LEAK_REPL_ENV
export LEAK_PAT_USER_EXTRA LEAK_PAT_BUNDLE_A LEAK_PAT_BUNDLE_B LEAK_PAT_BUNDLE_C LEAK_PAT_BUNDLE_D
export LEAK_PAT_BUNDLE_E LEAK_PAT_BUNDLE_UC LEAK_PAT_PARTITION LEAK_PAT_NODE LEAK_PAT_NODES
export LEAK_PAT_CPU_PART LEAK_PAT_CLUSTER_LOGIN LEAK_PAT_CLUSTER_HOST LEAK_PAT_FS LEAK_PAT_GROUP_FS
find "$ST" -type f ! -path '*/scripts/stage_push.sh' | while read -r f; do
  python3 "$D/redact_tree.py" "$f"
done
COMMIT_SP="$S/wt/push/results/scripts/stage_push.sh"
cp "$COMMIT_SP" "$ST/scripts/stage_push.sh"

find "$ST" -name 'cpu_job_*.log' -exec bash -c 'mv "$1" "$(dirname "$1")/cpu_job.log"' _ {} \; 2>/dev/null || true
find "$ST/partial-results" -type d -name tmp -exec rm -rf {} + 2>/dev/null || true
bad_names=
if [ "$LEAK_USER" != "<USER>" ]; then
  bad_names=$(find "$ST" -name "*${LEAK_USER}*" ! -path '*/tmp/*' 2>/dev/null | head -20 || true)
fi
if [ -n "$bad_names" ]; then
  echo "LEAK: suspicious file names:"
  echo "$bad_names"
  exit 1
fi
if [ "$LEAK_GROUP" != "<GROUP>" ]; then
  bad_names=$(find "$ST" -name "*${LEAK_GROUP}*" ! -path '*/tmp/*' 2>/dev/null | head -20 || true)
  if [ -n "$bad_names" ]; then
    echo "LEAK: suspicious file names (group):"
    echo "$bad_names"
    exit 1
  fi
fi

: "${LEAK_GREP_I:?LEAK_GREP_I unset — source stage_push.local.sh}"
LEAK_RE="$LEAK_GREP_I"
if grep -rniE "$LEAK_RE" "$ST" >/tmp/stage_push_leak.txt 2>/dev/null; then
  echo "== LEAK: contents (must be empty) =="
  head -40 /tmp/stage_push_leak.txt
  exit 1
fi
if find "$ST" -printf '%P\n' | grep -niE "$LEAK_RE" >/tmp/stage_push_leak_paths.txt 2>/dev/null; then
  echo "== LEAK: relative paths under stage_out (must be empty) =="
  head -20 /tmp/stage_push_leak_paths.txt
  exit 1
fi

bash "$D/stage_push_validate.sh" "$ST"

echo "== leak check OK =="
find "$ST" -type f \( -name '*.png' -o -name '*.safetensors' -o -name '*.pt' -o -name '*.sqsh' \)
du -sh "$ST"
echo "stage_out ready: $ST"