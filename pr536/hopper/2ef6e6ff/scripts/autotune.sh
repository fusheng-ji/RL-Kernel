source <WORK>/pr536/env.sh
cd $P/head; export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3; O=$OUT/head/autotune; mkdir -p $O
export RL_KERNEL_AUTOTUNE_CACHE_DIR=$(mktemp -d)
git status --porcelain --untracked-files=no > $O/status-before.txt
$PY -m pytest -p no:cacheprovider -q -rs tests/platforms/test_persistent_autotuner.py tests/runtime/test_kernel_registry.py tests/ops/test_autotuning_gpu.py tests/ops/test_autotuning_candidates.py tests/benchmarks/test_autotuning_benchmark.py > $O/suite.log 2>&1; echo "suite rc=$?" > $O/rc.txt
export RL_KERNEL_AUTOTUNE_CACHE_DIR=$(mktemp -d)
$PY -m pytest -p no:cacheprovider -v -rs tests/platforms/test_persistent_autotuner.py -k out_of_memory > $O/oom-v.log 2>&1; echo "oom rc=$?" >> $O/rc.txt
git status --porcelain --untracked-files=no > $O/status-after.txt
