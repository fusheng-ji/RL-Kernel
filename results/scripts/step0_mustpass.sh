#!/usr/bin/env bash
# Must-pass for allocation A (run from login after step0 on both nodes). Sets gates/step0_A.
source "$(dirname "$0")/common.sh"; J=$1; fail=""
# 8 cards memory.used == 0 at start
for role in node-a node-b; do
  f=$OUT/checks/step0-$role/gpus.csv; [ -f "$f" ] || fail="$fail missing $f"
  awk -F, 'NR>1 {gsub(/[^0-9]/,"",$4); if ($4+0 != 0) exit 1}' "$f" 2>/dev/null || fail="$fail $role memory.used"
done
# shared RW
for role in node-a node-b; do
  grep -q "^ok-$role" $SCHED/checks/rw-$role.txt 2>/dev/null || fail="$fail rw-$role"
done
# Hub or offline FA3 cache
hub_ok=0
grep -q '200' $OUT/checks/step0-node-a/hub.txt 2>/dev/null && hub_ok=1
grep -q '200' $OUT/checks/step0-node-b/hub.txt 2>/dev/null && hub_ok=1
cache=$SCHED/verlomni/hf/hub/kernels--kernels-community--flash-attn3/snapshots/7cb368cf8278b583132eb72cbf312d54586df2e2
[ -d "$cache" ] && hub_ok=1
[ $hub_ok = 1 ] || fail="$fail hub_and_no_fa3_cache"
# vime local (only lane 3): if /tmp or VIME_LOCAL has < 20G avail, set gate vime_local
VIME_LOCAL=${VIME_LOCAL:-/tmp}
av=$(df -BM "$VIME_LOCAL" 2>/dev/null | awk 'NR==2{gsub(/M/,"",$4); print $4}')
if [ -n "$av" ] && [ "$av" -lt 20480 ] 2>/dev/null; then
  gate_set vime_local "FAIL avail_MB=$av"
else gate_set vime_local OK; fi
if [ -n "$fail" ]; then gate_set step0_A "FAIL $fail"; echo "step0 must-pass FAIL:$fail"; exit 1; fi
gate_set step0_A OK; echo "step0 must-pass OK"