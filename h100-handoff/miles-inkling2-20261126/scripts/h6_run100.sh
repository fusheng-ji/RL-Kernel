#!/bin/bash
# H6: 100 steps with --dump-details and HF exports after rollouts 32/65/98 (+ the forced final one, 99).
# node_entry.sh runs the live-engine checksum sidecar on both nodes (after each export's weight sync).
set -x
source $THIS/scripts/lib_train.sh
O=$RUN/run; mkdir -p $O/hf
run_miles 100 "--dump-details $O/dump --save $O/ckpt --save-interval 33 --save-hf $O/hf/{rollout_id}" run; rc=$?
du -sh $O/hf/* $O/dump
exit $rc
