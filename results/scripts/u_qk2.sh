#!/usr/bin/env bash
# QK2 (single-card README "QK2"): a. real-weights QK tests (must pass, not skip); b. ws1 evidence test on head and base.
source "$(dirname "$0")/common.sh"; O=$OUT/QK2; mkdir -p $O
gate_wait build_WQ 3600 && gate_wait build_WB 3600 || exit 1
preflight $O $E/bin/python || exit 1
cp $OUT/build/WQ/build.log $O/build.log; cp $OUT/build/WB/build.log $O/build-base.log
cd $WQ; RL_KERNEL_H3_WEIGHTS=$WEIGHTS PYTHONPATH=$WQ:$PROVIDER/src CUDA_VISIBLE_DEVICES=$GPU $E/bin/python -m pytest -v -p no:cacheprovider -rfs \
  tests/models/minimax_h3/test_h3_qk_rmsnorm.py -k "test_block_0s_weights or test_the_strict_block_and_refiner_bind_the_row" > $O/qk-weights.txt 2>&1
for W in $WQ $WB; do n=$(basename $W); cd $W
  $E/bin/python -c "from rl_engine.backends.extension import _C; print(_C.__file__)" > $O/c-file-$n.txt
  PYTHONPATH=$W:$PROVIDER/src CUDA_VISIBLE_DEVICES=$GPU $E/bin/python -m pytest -v -p no:cacheprovider -rfs \
    tests/validation/operators/test_ws1_candidate_evidence.py::test_ws1_cuda_and_triton_candidate_runtime_provenance > $O/ws1-evidence-$n.txt 2>&1
done
for W in $WQ $WB; do tree_status $W $O/status-after-$(basename $W).txt; done
