#!/bin/bash
# H5: clean recipe run, no dump and no weight check. args: <num_rollout>
set -x
source $THIS/scripts/lib_train.sh
run_miles $1 ""
