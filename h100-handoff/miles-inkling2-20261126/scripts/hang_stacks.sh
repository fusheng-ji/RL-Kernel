#!/bin/bash
# Hang capture for the first training step (H2, H3), one per node. Once the stage's train.log ($RUN/train.log, or
# $RUN/<probe>/train.log for H2) shows "Timer log_probs start", waits
# 240 s; if "Timer log_probs end" has not appeared, dumps the Python stack of every trainer actor on this node with
# py-spy (<WORK>/tools/py-spy, see H0), twice, 120 s apart. Without py-spy it records each thread's kernel wait channel.
# args: <stage run dir> <out dir> <node id>
D=$1; OUT=$2; N=$3
PYSPY=$WORK/tools/py-spy
mkdir -p "$OUT"
LOG=
until [ -n "$LOG" ]; do
  sleep 10
  LOG=$(grep -l "Timer log_probs start" "$D"/train.log "$D"/*/train.log 2>/dev/null | head -1)
done
echo "watching $LOG"
sleep 240
for round in 1 2; do
  grep -q "Timer log_probs end" "$LOG" 2>/dev/null && { echo "log_probs ended, no dump" > "$OUT/node${N}_none.txt"; exit 0; }
  for p in $(pgrep -f "^ray::TorchtitanTrainRayActor"); do
    f="$OUT/node${N}_r${round}_pid$p.txt"
    date -u +%FT%TZ > "$f"
    if [ -x "$PYSPY" ]; then
      "$PYSPY" dump --pid "$p" >> "$f" 2>&1
    else
      echo "py-spy not found at <WORK>/tools; kernel wait channel per thread:" >> "$f"
      for t in /proc/$p/task/*; do echo "$(basename "$t") $(cat "$t/comm" 2>/dev/null) $(cat "$t/wchan" 2>/dev/null)"; done >> "$f"
    fi
  done
  sleep 120
done
