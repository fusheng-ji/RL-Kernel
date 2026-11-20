source <WORK>/pr536/env.sh
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3
T="tests/ops/gemm/test_det_gemm.py::test_target_shapes_invariance"
for n in head main sm90; do
  cd $P/$n; O=$OUT/$n/crash; mkdir -p $O
  git status --porcelain --untracked-files=no > $O/status-before.txt
  $PY -c "import rl_engine._C as c; print(c.__file__)" > $O/c-file.txt 2>&1
  RL_KERNEL_AUTOTUNE_CACHE_DIR=$(mktemp -d) CUDA_LAUNCH_BLOCKING=1 $PY -m pytest -p no:cacheprovider -v -rs "$T" > $O/crash-test.log 2>&1; echo "crash-test rc=$?" > $O/rc.txt
  if [ $n = sm90 ]; then
    RL_KERNEL_AUTOTUNE_CACHE_DIR=$(mktemp -d) CUDA_LAUNCH_BLOCKING=1 $PY -m pytest -p no:cacheprovider -v -rs tests/ops/gemm/test_det_gemm.py > $O/whole-det-gemm.log 2>&1; echo "whole-file rc=$?" >> $O/rc.txt
    RL_KERNEL_AUTOTUNE_CACHE_DIR=$(mktemp -d) $PY -m pytest -p no:cacheprovider -q -rs tests/ops/gemm/test_det_gemm.py > $O/whole-det-gemm-noblocking.log 2>&1; echo "whole-file-no-launch-blocking rc=$?" >> $O/rc.txt
  fi
  git status --porcelain --untracked-files=no > $O/status-after.txt
done; echo ALLDONE > $OUT/crash-done.txt
