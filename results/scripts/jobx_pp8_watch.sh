#!/usr/bin/env bash
# Report as soon as P3 row pp8-s925 appears in rows.txt (exit code may be non-zero).
D=$(dirname "$0"); source "$D/common.sh"
F=$OUT/pipeline/P3/rows.txt
log "jobx pp8 watch started"
while :; do
  if [ -f "$F" ] && grep -qE '^pp8-s925 exit=' "$F"; then
    line=$(grep -E '^pp8-s925 exit=' "$F" | tail -1)
    log "P3 pp8-s925 result: $line"
  echo "$line" > "$OUT/checks/p3-pp8-result.txt"
    exit 0
  fi
  pgrep -f 'u_jobx_b1_chain.sh' >/dev/null || squeue -h -s -j "$JA" -o '%i' 2>/dev/null | grep -q jobxP3 || {
    [ -f "$F" ] && grep -q pp8-s925 "$F" && exit 0
    [ -f "$F" ] && ! pgrep -f u_pipeline.sh >/dev/null && { log "pp8 watch: driver ended without pp8 line"; exit 0; }
  }
  sleep 30
done