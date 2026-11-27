#!/usr/bin/env bash
# P1/P2 rerun, then F3 rerun on an idle node-a. First run (OOM at update_weights_from_ipc on GPU 1) is kept in not-green-attempts.
D=$(dirname "$0"); source $D/common.sh
# P1/P2 rerun first (attempt 1 failed: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True broke cudaIpcGetMemHandle in DeterministicCollective; P2 also srun CPU request)
PA=$OUT/not-green-attempts/pipeline_P1P2_attempt1_expandable_segments_and_srun_cpus; mkdir -p $PA
for P in P1 P2; do [ -d $OUT/pipeline/$P ] && mv $OUT/pipeline/$P $PA/; done
echo "P1: 7 failed (CUDA invalid argument in DeterministicCollective IPC) with expandable_segments:True set by u_pipeline.sh; repro: rl_engine._C.deterministic_collective_ipc_meta fails with it and works without. P2: every row 'srun: More processors requested than permitted' (srun inherited 64 CPUs per task from the node step). Fixed in u_pipeline.sh: no expandable_segments, srun -c 16/8." > $PA/NOTE.txt
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash $D/u_pipeline.sh P1; bash $D/u_pipeline.sh P2
A=$OUT/not-green-attempts/verlomni_F3_attempt1_during_other_runs; mkdir -p $A
for r in w1-fa3-cap0-r0 w1-sdpa-cap0-r0; do [ -d $OUT/verlomni/runs/$r ] && mv $OUT/verlomni/runs/$r $A/; done
cp $OUT/verlomni/summary-w1.json $OUT/verlomni/analyze-w1.log $A/ 2>/dev/null
echo "First F3 (node-a, other work not running on node-a GPUs; node-b/B2 not yet started): both arms OOM in update_weights_from_ipc, GPU1 ~95 GB. Rerun on a fully idle job." > $A/NOTE.txt
nvidia-smi --query-gpu=index,memory.used --format=csv > $OUT/verlomni/F3-rerun-gpu-idle-check.txt
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash $D/u_verlomni.sh F3
ok=$(python3 - $OUT/verlomni/summary-w1.json <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(" ".join(k + ":" + [s for s in v["summary"] if "train exit" in s][0] for k, v in d["runs"].items()))
except Exception as e:
    print("no summary", e)
PY
)
gate_set verlomni_F3 "done (result, not pass/fail) $ok"
