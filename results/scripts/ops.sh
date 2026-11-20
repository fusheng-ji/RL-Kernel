source <WORK>/pr536/env.sh
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3
for n in head main; do for A in 0 1; do
  cd $P/$n; O=$OUT/$n/ops-autotune$A; mkdir -p $O; : > $O/rc.txt
  git status --porcelain --untracked-files=no > $O/status-before.txt
  for f in $(find tests/ops -name 'test_*.py' | sort); do
    RL_KERNEL_AUTOTUNE=$A RL_KERNEL_AUTOTUNE_CACHE_DIR=$(mktemp -d) timeout 7200 $PY -m pytest -p no:cacheprovider -q -rs "$f" > "$O/$(echo "$f" | tr / _).log" 2>&1
    echo "$f rc=$?" >> $O/rc.txt
  done
  git status --porcelain --untracked-files=no > $O/status-after.txt
  echo "$n A=$A done $(date +%T)" >> $OUT/ops-progress.txt
done; done
echo ALLDONE >> $OUT/ops-progress.txt
