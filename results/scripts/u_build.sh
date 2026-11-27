#!/usr/bin/env bash
# u_build.sh W10|WQ|WB|WP : compile sm90 in the named checkout on the card this step got. One step per checkout.
source "$(dirname "$0")/common.sh"; K=$1; C=$WT/$K; O=$OUT/build/$K; mkdir -p $O
preflight $O || exit 1
tree_status $C $O/status-before.txt || { echo "dirty $K"; exit 1; }
export MAX_JOBS=${MAX_JOBS:-8}
cd $C
case $K in
  W10|WP) RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 $E1/bin/python setup.py build_ext --inplace > $O/build.log 2>&1; rc=$?;;
  WQ|WB)  RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 KERNEL_ALIGN_DET_GEMM_SM90=1 $E/bin/python setup.py build_ext --inplace > $O/build.log 2>&1; rc=$?;;
esac
echo "build_exit=$rc" >> $O/build.log
cfile $O $C $E1 $K || rc=9
[ $K = W10 ] && PYTHONPATH=$C $V/bin/python -c "import os, rl_engine._C as c, rl_engine._H3_C as h, triton; p=os.path.realpath; print(p(c.__file__).startswith(p('$C')), h.h3_attention_build_info()['profile'], triton.__version__)" > $O/ext.txt 2>&1
tree_status $C $O/status-after.txt || rc=8
[ $rc = 0 ] && gate_set build_$K OK || gate_set build_$K "FAIL rc=$rc"
