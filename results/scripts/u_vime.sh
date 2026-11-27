#!/usr/bin/env bash
# vime V0..V4 by running the handoff sbatch files inside the allocation (alloc_driver / gpu_step), not with sbatch.
# usage: u_vime.sh V0|V2|V3|V4 ; env: JOB, NODEA (first node of JOB)   gates: vime_V0 vime_V2
source "$(dirname "$0")/common.sh"
export H=$WT/h_vi/h100-handoff/vime-dsv4-20261126 W=$SCHED/vime SCRATCH=${VIME_LOCAL:-/tmp} HF_HUB_OFFLINE=1
export SLURM_CPUS_PER_TASK=${CPT:-32}; S=$W/runs
case $1 in
 V0) bash $H/build.sbatch > $OUT/vime/V0.log 2>&1; rc=$?                      # one card, 32 CPU instead of 48 (64 CPU per node)
     cp $W/wheels-sm90/env-freeze-sm90.txt $OUT/vime/ 2>/dev/null; cp $W/logs/image.sha256 $OUT/vime/ 2>/dev/null
     if [ $rc = 0 ] && grep -q BUILD_DONE $OUT/vime/V0.log && ! grep -q STEP_FAIL $OUT/vime/V0.log; then gate_set vime_V0 OK; else gate_set vime_V0 "FAIL rc=$rc (STEP_FAIL or no BUILD_DONE)"; fi;;
 V2) gate_ok vime_V0 || exit 1; bash $H/smoke.sbatch > $OUT/vime/V2.log 2>&1; rc=$?
     if [ $rc = 0 ]; then gate_set vime_V2 OK; else gate_set vime_V2 "FAIL rc=$rc"; exit $rc; fi
     # cuDNN DSA failure is not a stop: V3/V4 then run with DSA_BACKEND=none
     python3 - $S/smoke/preflight.json <<'PY' && echo cudnn_dsa_ok > $GATES/vime_dsa || echo "DSA_BACKEND=none" > $GATES/vime_dsa
import json, sys
try: d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("cudnn_dsa", {}).get("wrapper") is True else 1)
except Exception: sys.exit(1)
PY
     ;;
 V3|V4) gate_ok vime_V0 && gate_wait vime_V2 3600 || exit 1
     grep -q none $GATES/vime_dsa 2>/dev/null && export DSA_BACKEND=none
     export SLURM_GRES=gpu:4 SLURM_OVERLAP=1
     if [ $1 = V3 ]; then bash $H/core.sbatch > $OUT/vime/V3.log 2>&1; python3 $H/analyze.py $S/core > $OUT/vime/V3-summary.json
     else bash $H/drift.sbatch > $OUT/vime/V4.log 2>&1; python3 $H/analyze.py $S/drift > $OUT/vime/V4-summary.json; fi;;
esac
