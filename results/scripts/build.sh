source <WORK>/pr536/env.sh
n=$1; shift; cd $P/$n; mkdir -p $OUT/$n
( echo "== $n"; git rev-parse HEAD; git status --porcelain --untracked-files=no | wc -l ) > $OUT/$n/pre-build-status.txt
env "$@" TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 $PY setup.py build_ext --inplace > $OUT/$n/build.log 2>&1; echo "$n build rc=$?"
PYTHONPATH=. $PY -c "import rl_engine._C as c; print(c.__file__)" > $OUT/$n/c-file.txt 2>&1; cat $OUT/$n/c-file.txt | tail -1
