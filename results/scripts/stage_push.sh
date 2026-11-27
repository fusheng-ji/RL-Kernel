#!/usr/bin/env bash
# Build redacted stage tree for handoff results branches. Does not push.
# Usage: SCHED_ROOT=/path/to/sched bash stage_push.sh
# Committed copy: site-specific patterns are shown as <PLACEHOLDERS>; the copy that runs holds the real values.
set -euo pipefail
S=${SCHED_ROOT:-${SCRATCH_ROOT:?}/sched_20261127}
ST=$S/stage_out
LEAK_USER=${LEAK_USER:-<USER>}
LEAK_JOB1=${LEAK_JOB1:-<JOB_1>}
LEAK_JOB2=${LEAK_JOB2:-<JOB_2>}

rm -rf "$ST"
mkdir -p "$ST"/{scripts,checks,not-green-attempts,partial-results}
cp "$S/stage/PLAN.md" "$ST/PLAN.md"
[ -f "$S/stage/RUN_SUMMARY.md" ] && cp "$S/stage/RUN_SUMMARY.md" "$ST/RUN_SUMMARY.md"
cp "$S/scripts"/*.sh "$S/scripts"/*.sbatch "$ST/scripts/" 2>/dev/null || true
cp "$S/scripts/stage_push.sh" "$ST/scripts/" 2>/dev/null || true
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
cp "$S/not-green-attempts"/* "$ST/not-green-attempts/" 2>/dev/null || true

for f in prep-verify.txt test_only_N2_exclusive_18h.txt miles_dl.log vime_prep.log verlomni_prep.log cpu_job.log sqsh-superblock.txt monitor.log; do
  [ -f "$S/checks/$f" ] && cp "$S/checks/$f" "$ST/checks/"
done
cp "$S/areal-release/areal-ropefusion-placement-run/NOTES.md" "$ST/checks/areal-placement-NOTES.md" 2>/dev/null || true
cp "$S/not-green-attempts"/* "$ST/not-green-attempts/" 2>/dev/null || true
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
export LEAK_USER LEAK_JOB1 LEAK_JOB2
find "$ST" -type f | while read -r f; do
  python3 - "$f" <<'PY'
import os, re, sys

p = sys.argv[1]
s = open(p, errors="replace").read()
u = os.environ.get("LEAK_USER", "<USER>")
j1 = os.environ.get("LEAK_JOB1", "<JOB_1>")
j2 = os.environ.get("LEAK_JOB2", "<JOB_2>")

# Order: specific before general (same table for all result branches).
R = [
    # work tree root
    (rf"<FS>", "<WORK>"),
    (rf"<FS>", "<SCRATCH>"),
    (r"<PROJECT_FS>"']*", "<PROJECT_FS>"),
    (rf"<FS>", "<HOME>"),
    # scratch env vars before bare SCRATCH_ROOT
    (r"\$SCRATCH_ROOT", "$SCRATCH_ROOT"),
    (r"\b<SCRATCH_VAR>\b", "SCRATCH_ROOT"),
    # venv names before generic _venv grep would fire in output
    (r"<E>", "<E>"),
    (r"<V>", "<V>"),
    (r"<E2>", "<E2>"),
    (r"<ENV>\w*", "<ENV>"),
    # AReaL bundle dir + variable (before generic <SITE>)
    (r"\$BUNDLE/<BUNDLE_SCRIPTS_DIR>\b", "$BUNDLE/<BUNDLE_SCRIPTS_DIR>"),
    (r"\$B/<BUNDLE_SCRIPTS_DIR>\b", "$B/<BUNDLE_SCRIPTS_DIR>"),
    (r"/<BUNDLE_SCRIPTS_DIR>\b", "/<BUNDLE_SCRIPTS_DIR>"),
    (r"/<BUNDLE_SCRIPTS_DIR>/", "/<BUNDLE_SCRIPTS_DIR>/"),
    (r"\b<SITE>/", "<BUNDLE_SCRIPTS_DIR>/"),
    (r"\b<SITE_UC>\b", "BUNDLE_SCRIPTS"),
    # jobs
    (rf"\b{re.escape(j1)}\b", "<JOB_1>"),
    (rf"\b{re.escape(j2)}\b", "<JOB_2>"),
    (r"\b59\d{5}\b", "<JOBID>"),
    (r"cpu_job_\d+\.log", "cpu_job.log"),
    # partition / nodes / qos
    (r"<PARTITION>", "<PARTITION>"),
    (r"<CLUSTER>\d+", "node-x"),
    (r"<CPU_PARTITION>", "<CPU_PARTITION>"),
    (r"cpu-0\d\d", "node-c"),
    (r"--qos=\w+", "--qos=<QOS>"),
    (r"<CLUSTER>\[[0-9,]+\]", "<NODES>"),
    # identity
    (re.escape(u), "<USER>"),
    (r"<GROUP>[\w-]*", "<GROUP>"),
    (r"<USER>|<LOCAL_USER>@\S+", "<USER>"),
    # network / hardware ids
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>"),
    (r"GPU-[0-9a-f]{8}-[0-9a-f-]{27}", "<GPU-UUID>"),
    # cluster hostnames only (not DRIVER_MODE=login)
    (r"\blogin[-.][\w.]+", "<CLUSTER>"),
    (r"\b<CLUSTER_PREFIX>[-\w.]+", "<CLUSTER>"),
    # remaining <FS> paths
    (r"<FS>"']*", "<FS>"),
]

for a, b in R:
    s = re.sub(a, b, s)
open(p, "w").write(s)
PY
done

find "$ST" -name 'cpu_job_*.log' -exec bash -c 'mv "$1" "$(dirname "$1")/cpu_job.log"' _ {} \; 2>/dev/null || true

find "$ST/partial-results" -type d -name tmp -exec rm -rf {} + 2>/dev/null || true
bad_names=$(find "$ST" \( -name "*${LEAK_USER}*" -o -name '*<GROUP>*' \) ! -path '*/tmp/*' 2>/dev/null | head -20 || true)
if [ -n "$bad_names" ]; then
  echo "LEAK: suspicious file names:"
  echo "$bad_names"
  exit 1
fi

LEAK_RE="<FS_PREFIX>|<CLUSTER_PREFIX>|<SITE>|${LEAK_USER}|${LEAK_JOB1}|${LEAK_JOB2}|_venv"
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

echo "== leak check OK =="
find "$ST" -type f \( -name '*.png' -o -name '*.safetensors' -o -name '*.pt' -o -name '*.sqsh' \)
du -sh "$ST"
echo "stage_out ready: $ST"