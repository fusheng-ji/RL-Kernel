#!/bin/bash
# H2 memory probe: one real recipe step (rollout, train, weight sync) at one SGLang setting. args: <name> <SG_TP> <SG_MEM>
# An out-of-memory error is a probe result: the stage exits 0 and train.log keeps the error.
set -x
export SG_TP=$2 SG_MEM=$3
source $THIS/scripts/lib_train.sh
SG_TP=$2; SG_MEM=$3
RUN_TIMEOUT=3600 run_miles 1 "" $1; echo "$1 rc=$?"
exit 0
