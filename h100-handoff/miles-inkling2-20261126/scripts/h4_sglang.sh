#!/bin/bash
# H4b (one node): SGLang self-consistency on H3 rollout 0. args: <H3 dump dir>
set -x
[ -f "$WORK/h100.env" ] && source "$WORK/h100.env"
cd /root
python3 $THIS/scripts/sglang_selfcheck.py --dump $1 --out $RUN/sglang_default.json --log $RUN/server_default.log
python3 $THIS/scripts/sglang_selfcheck.py --dump $1 --out $RUN/sglang_deterministic.json --log $RUN/server_det.log --deterministic
rc=0; for f in default deterministic; do [ -s $RUN/sglang_$f.json ] || rc=1; done
exit $rc
