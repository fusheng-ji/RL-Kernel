#!/bin/bash
# H7b: trainer with the ORIGINAL weights scores late rollout data (one train-only step; old_log_probs are before the update).
# args: <H6 dump dir> <rollout id> ...
set -x
source $THIS/scripts/lib_train.sh
D=$1; shift; rc=0
for k in "$@"; do
  mkdir -p $RUN/data_$k && ln -sf $D/rollout_data/$k.pt $RUN/data_$k/0.pt
  run_miles 1 "--load-debug-rollout-data $RUN/data_$k/{rollout_id}.pt --dump-details $RUN/r$k/dump" r$k || rc=1
done
exit $rc
