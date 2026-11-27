#!/usr/bin/env bash
# JOB_Y: H0b + round1 archive, then areal_done → exclusive 2-node driver → b3 (+ H3 if pass).
D=$(dirname "$0"); source "$D/common.sh"
J=${1:-$JB}
MI=$SCHED/miles
log "miles b3 waiter started job=$J"
round1_archive() {
  local d="$MI/not-green-attempts/round1"
  mkdir -p "$d"
  for r in "$MI/runs"/h2_memprobe-{a,b,c,d}; do
    [ -d "$r" ] && mv "$r" "$d/"
  done
}
round1_archive
TAG=b3
while :; do
  spysha=$MI/runs/h0/py-spy.sha256
  if [ ! -f "$spysha" ] || ! grep -q OK "$spysha" || ! grep -q 'py-spy 0.4.0' "$spysha"; then
    sleep 120
    continue
  fi
  gate_ok areal_done || { sleep 120; continue; }
  busy=$(squeue -h -s -j "$J" -o '%i' 2>/dev/null | grep -v '\.batch$' | grep -c . || true)
  [ "$busy" = 0 ] || { sleep 120; continue; }
  for n in $(nodes_of "$J"); do
    other=$(squeue -h -w "$n" -o '%i' 2>/dev/null | grep -v "^$J" | grep -c . || true)
    [ "$other" = 0 ] || { log "STOP: nodes not exclusive for miles b3"; exit 1; }
  done
  break
done
NA=$(nodes_of "$J" | sed -n 1p)
while :; do
  driver_2node_sync "$J" "$NA" "3:30:00" "milesB3_${TAG}" \
    "export TAG=$TAG SLURM_JOB_ID=$J; bash $D/u_miles_b3.sh $J"
  b3rc=$?
  if [ -f "$GATES/miles_H2_busy" ] && grep -q 'BUSY rc=96' "$GATES/miles_H2_busy"; then
    case $TAG in b3) TAG=b3a ;; b3a) TAG=b3b ;; b3b) TAG=b3c ;; *) TAG=b3_retry ;; esac
    log "miles b3 busy rc=96 → retry TAG=$TAG"
    sleep 300
    continue
  fi
  if [ -f "$GATES/miles_H3" ] && grep -q 'done rc=0' "$GATES/miles_H3"; then
    log "miles round3: H3 done — stopping for review"
    exit 0
  fi
  if [ $b3rc -ne 0 ] && [ $b3rc -ne 96 ]; then
    log "STOP: miles H2 b3 failed TAG=$TAG — see train.log stacks fr ray_logs"
    exit 1
  fi
  break
done