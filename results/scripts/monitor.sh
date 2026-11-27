#!/bin/bash
# Poll JA/JB; on RUNNING call on_job_start.sh once per job.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
D=$(dirname "$0")
prev=""; started_a=0; started_b=0
while true; do
  cur=$(squeue -h -j ${JA},${JB} -o '%i %T %N' 2>/dev/null | sort | tr '\n' ';')
  if [ "$cur" != "$prev" ]; then echo "$(date -u +%H:%M) JOBSTATE $cur" >> "$SCHED/checks/monitor.log"; prev="$cur"; fi
  sa=$(squeue -h -j "$JA" -o %T 2>/dev/null)
  sb=$(squeue -h -j "$JB" -o %T 2>/dev/null)
  if [ "$sa" = RUNNING ] && [ "$started_a" = 0 ]; then setsid nohup bash "$D/on_job_start.sh" "$JA" first & disown; started_a=1; fi
  if [ "$sb" = RUNNING ] && [ "$started_b" = 0 ]; then setsid nohup bash "$D/on_job_start.sh" "$JB" second & disown; started_b=1; fi
  sleep 300
done