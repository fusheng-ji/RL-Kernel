#!/usr/bin/env bash
# verl-omni F0..F3 on a whole node (node-a of allocation A: clean timing). usage: u_verlomni.sh [F3]
# Gate: F0 same_kernel_file must be true and both repeats bitwise, else stop here (gate verlomni_F0 = FAIL).
source "$(dirname "$0")/common.sh"; HV=$WT/h_vo/h100-handoff/verl-omni-fa3-20261124; V0=$SCHED/verlomni
export E2 VERL_OMNI=$V0/verl-omni CKPT=$SCRATCH_ROOT/h3ckpt/hf_full REWARDS=$V0/rewards OVERLAY=$V0/overlay OUT=$OUT/verlomni
mkdir -p $OUT/F0 $OUT/not-green-attempts; O=$OUT
export HF_HOME=$V0/hf SHORT_TMP=${SHORT_TMP:-/tmp}
git -C $VERL_OMNI status --porcelain --untracked-files=no > $OUT/F0/status-before.txt; [ -s $OUT/F0/status-before.txt ] && { gate_set verlomni_F0 "FAIL dirty"; exit 1; }
nvidia-smi > $OUT/F0/nvidia-smi-start.txt
case "${1:-}" in F3) ;; *)
  cd $OUT && CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=$OVERLAY $E2/bin/python $HV/preflight.py > $OUT/F0/preflight.json 2> $OUT/F0/preflight.err
  $E2/bin/python -c "import importlib.metadata as m; [print(p+'=='+m.version(p)) for p in ['torch','vllm','vllm-omni','verl','diffusers','transformers','kernels','huggingface-hub']]" > $OUT/F0/pip-list.txt
  python3 - $OUT/F0/preflight.json <<'PY' && gate_set verlomni_F0 OK || { gate_set verlomni_F0 "FAIL F0 gate"; exit 1; }
import json, sys
d = json.load(open(sys.argv[1])); f = d.get("fa3", {})
ok = d.get("same_kernel_file") is True and f.get("actor_repeat_bitwise") is True and f.get("rollout_repeat_bitwise") is True
print("F0 gate:", ok); sys.exit(0 if ok else 1)
PY
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  run() { ARM=$1 WL=$2 CAP=$3 STEPS=$4 TAG=$5 bash $HV/run_arm.sh; echo "arm=$1 wl=$2 cap=$3 tag=$5 exit=$?" >> $OUT/steps.txt; }
  run sdpa w0 0 4 r0; run sdpa w0 1 2 r0                                       # F1
  run fa3 w0 0 4 r0; run fa3 w0 0 4 r1; run fa3 w0 1 2 r0; run fa3 w0 1 2 r1   # F2
  (cd $OUT/runs && $E2/bin/python $HV/analyze.py w0-*-cap0-* w0-*-cap1-* --repeat w0-fa3-cap1-r0 w0-fa3-cap1-r1 --out $OUT/summary-w0.json) > $OUT/analyze-w0.log 2>&1
  gate_set verlomni_F2 done;;
esac
if [ "${1:-}" = F3 ] || [ "${RUN_F3:-0}" = 1 ]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  ARM=fa3 WL=w1 CAP=0 STEPS=3 TAG=r0 bash $HV/run_arm.sh; ARM=sdpa WL=w1 CAP=0 STEPS=3 TAG=r0 bash $HV/run_arm.sh   # OOM is a result, no offload
  (cd $OUT/runs && $E2/bin/python $HV/analyze.py w1-* --out $OUT/summary-w1.json) > $OUT/analyze-w1.log 2>&1
fi
