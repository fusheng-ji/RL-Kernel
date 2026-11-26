#!/bin/bash
# H4a: fixed rollout data of H3 -> two trainer-only replays (repeatability), then the two consistency-option probes.
# args: <H3 dump dir>
set -x
source $THIS/scripts/lib_train.sh
DUMP=$1
for i in a b; do
  run_miles 3 "--load-debug-rollout-data $DUMP/rollout_data/{rollout_id}.pt --dump-details $RUN/rep_$i/dump" rep_$i; echo "rep_$i rc=$?"
done
# a refused option is a valid probe result; read train.log
RUN_TIMEOUT=2400 run_miles 1 "--use-rollout-routing-replay" probe_r3; echo "probe_r3 rc=$?"
RUN_TIMEOUT=2400 run_miles 1 "--true-on-policy-mode" probe_top; echo "probe_top rc=$?"
python3 $THIS/scripts/cmp_replay.py $RUN/rep_a/dump $RUN/rep_b/dump | tee $RUN/replay_bitwise.txt
exit 0
