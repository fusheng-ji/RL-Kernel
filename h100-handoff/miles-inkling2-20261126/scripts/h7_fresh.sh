#!/bin/bash
# H7a (one node): fresh SGLang (SG_TP GPUs) on a checkpoint: weights checksum, then prefill-score one rollout.
# args: "<model dir>:<rollout .pt>:<name>" ...
set -x
[ -f "$WORK/h100.env" ] && source "$WORK/h100.env"
cd /root; rc=0
for spec in "$@"; do
  IFS=: read -r M P N <<<"$spec"
  [ -s $RUN/$N.pt ] && continue
  python3 $THIS/scripts/score_rollout.py --model $M --rollout $P --out $RUN/$N.pt --log $RUN/$N.log --checksum $RUN/${N}_checksum.json || rc=1
  [ -s $RUN/$N.pt ] || rc=1
done
exit $rc
