#!/usr/bin/env bash
# Read-only progress poller: gates, steps, jobs, handoff milestones, periodic summary.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
INT=${1:-120}
SUMMARY_EVERY=${2:-10}   # every N polls (~20 min at 120s)
declare -A seen
last_steps=0; last_job=""; last_sacct=""
poll=0
handoff_once() {
  local id=$1 msg=$2
  [ -n "${seen[h_$id]+x}" ] && return 0
  seen[h_$id]=1
  echo "$(date -u +%FT%TZ) HANDOFF_READY $id: $msg"
}
check_handoff_milestones() {
  gate_ok step0_A && handoff_once step0 "step0 checks + nested_srun (schedule checks bundle)"
  gate_ok miles_H1 && handoff_once miles_H0_H1 "miles H0+H1 logs in out/miles (partial miles results branch)"
  if [ -f "$GATES/vime_V0" ] && grep -q FAIL "$GATES/vime_V0" 2>/dev/null; then
    handoff_once vime_V0_fail "vime V0 failure log out/vime/V0.log (partial, after node-b lane ends)"
  fi
  local T=$OUT/T10
  if [ -f "$T/a/run-edit.log" ] && [ -f "$T/a/run-t2i.log" ] && [ -f "$T/a/run-b.log" ]; then
    handoff_once T10_a "T10-a steps a–d outputs under out/T10/a"
  fi
  if [ -f "$T/edit/run-a.log" ] && [ -f "$T/edit/status-after.txt" ]; then
    handoff_once T10_edit "T10-edit under out/T10/edit"
  fi
  if [ -f "$OUT/QK2/qk-weights.txt" ] && [ -f "$OUT/QK2/ws1-evidence-WB.txt" ]; then
    handoff_once QK2 "QK2 weights + ws1 evidence in out/QK2"
  fi
  if [ -d "$OUT/pipeline/P1" ] && [ -f "$OUT/pipeline/P2/rows.txt" ]; then
    handoff_once pipeline_P1P2_nodea "pipeline P1+P2 on node-a (extend after B1 for P3/P4)"
  fi
  gate_ok nodea_done && handoff_once nodea_A "node-a A slice done → verl-omni + pipeline P1/P2 handback gate"
  gate_ok nodeb_done && handoff_once nodeb_A "node-b A slice done → T10 + QK2 + vime handback gate"
  gate_ok areal_done && handoff_once areal_C "C done → areal placement results"
  gate_ok miles_H5 && handoff_once miles_B2a "B2a done → miles H2–H5 bundle"
  for g in miles_H6 nodea_done; do :; done
  if [ -f "$GATES/miles_H5" ] && gate_ok miles_H5; then :; fi
}
print_remaining() {
  echo "$(date -u +%FT%TZ) REMAINING_SUMMARY begin"
  echo "  jobs: $(squeue -h -j "$JA,$JB" -o '%i=%T' 2>/dev/null | tr '\n' ' ')"
  echo "  nodea_done=$(gate_ok nodea_done && echo OK || echo pending) nodeb_done=$(gate_ok nodeb_done && echo OK || echo pending)"
  echo "  node-a: verlomni_F3=$(test -f "$GATES/verlomni_F3" && cat "$GATES/verlomni_F3" || echo pending) miles_H1=$(test -f "$GATES/miles_H1" && echo set || echo pending)"
  echo "  node-b: T10-a=$(test -f "$OUT/T10/a/run-b.log" && echo done || echo in_progress) vime_V0=$(test -f "$GATES/vime_V0" && head -c60 "$GATES/vime_V0" || pending)"
  echo "  not_started: B2a=$(gate_ok nodea_done && gate_ok nodeb_done || echo waiting_A) B2b=miles_H5 C=job_$JB B1=job_$JB+areal_done"
  echo "$(date -u +%FT%TZ) REMAINING_SUMMARY end"
}
echo "$(date -u +%FT%TZ) WATCH start interval=${INT}s summary_every=${SUMMARY_EVERY}"
while true; do
  poll=$((poll + 1))
  for g in "$GATES"/*; do
    [ -f "$g" ] || continue
    n=$(basename "$g"); c=$(cat "$g")
    if [ -z "${seen[$n]+x}" ] || [ "${seen[$n]}" != "$c" ]; then
      seen[$n]=$c
      case "$c" in *FAIL*) tag="GATE_FAIL";; *done*) tag="GATE_DONE";; *) tag="GATE";; esac
      echo "$(date -u +%FT%TZ) $tag $n: $c"
    fi
  done
  total=$(wc -l < "$OUT/steps.txt" 2>/dev/null || echo 0)
  if [ "$total" -gt "$last_steps" ]; then
    tail -n +$((last_steps + 1)) "$OUT/steps.txt" | while read -r l; do
      case "$l" in *STOP:*) echo "$(date -u +%FT%TZ) STOP_LOG $l";; *) echo "$(date -u +%FT%TZ) STEP $l";; esac
    done
    last_steps=$total
  fi
  job=$(squeue -h -j "$JA,$JB" -o '%i=%T' 2>/dev/null | sort | tr '\n' ' ')
  [ "$job" != "$last_job" ] && { echo "$(date -u +%FT%TZ) JOBS $job"; last_job=$job; }
  sac=$(sacct -j "$JA" -n -P --format=JobID,State 2>/dev/null | awk -F'|' '$0 ~ /\.(21|24|25)\|/ && $2 !~ /^RUNNING$/ {print}' | tr '\n' ' ')
  [ -n "$sac" ] && [ "$sac" != "$last_sacct" ] && { echo "$(date -u +%FT%TZ) STEPSTATE_ACTIVE $sac"; last_sacct=$sac; }
  check_handoff_milestones
  [ $((poll % SUMMARY_EVERY)) -eq 0 ] && print_remaining
  sleep "$INT"
done