#!/bin/bash
# V2, inside the container, one GPU. Same commands as the B200 round; OUT gets one log per item.
set -x; OUT=$1
export PYTHONNOUSERSITE=1 HOME=/tmp/home PYTHONDONTWRITEBYTECODE=1; mkdir -p $HOME
export XDG_CACHE_HOME=$W/jitcache VLLM_CACHE_ROOT=$W/jitcache/vllm TILELANG_CACHE_DIR=$W/jitcache/tilelang TRITON_CACHE_DIR=/tmp/triton-cache
# a. preflight (capability, cuDNN DSA, DeepGEMM scale format, FP8 grid) and the grad-norm kernel check
PYTHONPATH=$W/src/vime-fix-run:$W/src/Megatron-run:$H/probe V4_ALLOW_NUMPY2=1 python3 $H/preflight.py $W/ckpt/v4-4l > $OUT/preflight.json 2> $OUT/preflight.err; tail -3 $OUT/preflight.err
python3 $H/l2test.py > $OUT/l2test.log 2>&1
# b. the PR's own checks, on the clean PR head and clean Megatron
( cd $W/src/vime; export PYTHONPATH=$W/src/vime:$W/src/Megatron-LM DSV4_CHECKPOINT=$W/ckpt/v4-4l
  python3 -m pytest -q -p no:cacheprovider tests/test_deepseek_v4_weights.py tests/test_hf_to_megatron.py tests/test_deepseek_v4_native_mapping.py tests/test_deepseek_v4_routing_replay.py tests/test_stateless_adam.py tests/test_megatron_argument_validation.py tests/test_vllm_weight_checker.py -rs > $OUT/pr-pytest.log 2>&1; echo rc=$? >> $OUT/pr-pytest.log
  for a in "--layer-index 2" "--layer-index 3" "--num-layers 4"; do python3 tools/dsv4_checkpoint_layer_smoke.py $DSV4_CHECKPOINT $a > "$OUT/smoke${a// /_}.log" 2>&1; echo rc=$? >> "$OUT/smoke${a// /_}.log"; done
  python3 tools/dsv4_rollout_smoke.py $DSV4_CHECKPOINT --tensor-parallel-size 1 --return-routed-experts > $OUT/rollout_smoke.log 2>&1; echo rc=$? >> $OUT/rollout_smoke.log
  python3 tools/dsv4_runtime_checks.py $DSV4_CHECKPOINT > $OUT/runtime_checks.log 2>&1; echo rc=$? >> $OUT/runtime_checks.log
  git status --porcelain --untracked-files=no > $OUT/pr-tree-status.txt )
# c. unit tests on the fix branch (+ 0002) with patched Megatron
( cd $W/src/vime-fix-run; export PYTHONPATH=$W/src/vime-fix-run:$W/src/Megatron-run V4_ALLOW_NUMPY2=1
  python3 -m pytest -q -p no:cacheprovider tests/test_vllm_cli_args_compat.py tests/test_optimizer_grad_norm_guards.py tests/test_deepseek_v4_weights.py tests/test_hf_to_megatron.py tests/test_deepseek_v4_routing_replay.py tests/test_stateless_adam.py tests/test_megatron_argument_validation.py tests/test_vllm_weight_checker.py -rs > $OUT/fix-pytest.log 2>&1; echo rc=$? >> $OUT/fix-pytest.log )
python3 -m pip list --format=freeze > $OUT/pip-list.txt 2>/dev/null
for f in $OUT/*.log; do echo "== $(basename $f)"; tail -3 $f; done
