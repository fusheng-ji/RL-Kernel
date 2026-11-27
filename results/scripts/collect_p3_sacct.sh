#!/usr/bin/env bash
# Write sacct inner steps for current job P3 window to checks/p3-step-cpus.txt (caller redacts via stage_push).
D=$(dirname "$0"); source "$D/common.sh"
J=${1:-$JA}
_lp="${D}/stage_push.local.sh"
[ -f "$_lp" ] && source "$_lp"
HA=${LEAK_HOST_A:-node-a}; HB=${LEAK_HOST_B:-node-b}
O=$SCHED/checks/p3-step-cpus.txt
mkdir -p "$SCHED/checks"
{
  echo "# P3 pipeline inner steps job $J ($(date -u +%FT%TZ))"
  sacct -j "$J" --format=JobID,Start,End,AllocCPUS,NNodes,NodeList -P \
    | sed -e "s/${LEAK_HOST_A:-node-x}/node-a/g" \
          -e "s/${LEAK_HOST_B:-node-x}/node-b/g" \
          -e 's/<CLUSTER>\[[^]]*\]/node-a+node-b/g' \
    | grep -E "^${J}\." | sed "s/${J}/<JOB_1>/g"
} > "$O"
log "wrote $O"