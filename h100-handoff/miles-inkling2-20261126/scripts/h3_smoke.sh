#!/bin/bash
# H3: 3 steps with the bitwise weight-sync check and full dumps (the setting chosen in H2, from $WORK/h100.env).
set -x
source $THIS/scripts/lib_train.sh
run_miles 3 "--check-weight-update-equal --check-weight-update-skip-list mlp.gate._one --dump-details $RUN/dump"
rc=$?
grep -nE 'perf [0-9]+:|rollout/raw_reward|train_rollout_logprob_abs_diff|Error|Traceback|weight.*equal' $RUN/train.log | cut -c1-400 | tail -60
du -sh $RUN/dump
exit $rc
