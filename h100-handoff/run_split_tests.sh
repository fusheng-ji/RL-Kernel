#!/usr/bin/env bash
# Hopper (H100) runs for the split branches 539a-c (MiniMax-H3) and 543a-c (Qwen3-Next).
#
#   REPO=<clone of the fork> WEIGHTS=<dir with h3_conditioning.safetensors + h3_block0.safetensors> \
#   [PY=python] [WT=<worktree parent>] [EXTRA_PYTHONPATH=<overlay with sglang + matplotlib>] \
#   [SKIP_BUILD=1] [MAX_JOBS=32] bash run_split_tests.sh
#
# EXTRA_PYTHONPATH may also hold megatron-core (tested with 0.16.1, installed with `pip install --no-deps --target`);
# without it the 4 megatron tests in 543b/543c skip.
#
# Needs ninja on PATH (about 1 minute per build; about 15 minutes without it) and a CUDA toolkit
# matching the installed torch. Results and logs go to $WT/results.
set -u
REPO=${REPO:?set REPO to a clone of the fork}
WEIGHTS=${WEIGHTS:?set WEIGHTS to the H3 weights dir}
PY=${PY:-python}
WT=${WT:-$(dirname "$REPO")/rl-kernel-h100-split}
OUT=${OUT:-$WT/results}
EXTRA=${EXTRA_PYTHONPATH:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$WT" "$OUT"

H3=(feat/h3-one-block-replay feat/h3-full-model-qualification feat/h3-generation-gate)
QN=(feat/ws1-qwen3-next-chain-foundation feat/ws1-qwen3-next-chain-training-l2 feat/ws1-qwen3-next-chain-rollout-runners)

{
  echo "host: $(hostname)  date: $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv,noheader
  $PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
  echo "visible GPUs: $(nvidia-smi -L | wc -l)"
} | tee "$OUT/env.txt"

for b in "${H3[@]}" "${QN[@]}"; do git -C "$REPO" fetch -q origin "refs/heads/$b:refs/remotes/origin/$b" || { echo "fetch failed for $b"; exit 2; }; done

wt() { # branch -> worktree path (created from the fetched remote branch)
  local d="$WT/$(echo "$1" | tr '/' '_')"
  [ -d "$d" ] || git -C "$REPO" worktree add -q -B "$1" "$d" "origin/$1" || exit 2
  echo "$d"
}

build() { # dir strict(0|1)
  [ "${SKIP_BUILD:-0}" = 1 ] && return 0
  ( cd "$1" && RL_KERNEL_H3_STRICT=$2 MAX_JOBS=${MAX_JOBS:-32} $PY setup.py build_ext --inplace > "$OUT/build_$(basename "$1").log" 2>&1 ) \
    || { echo "BUILD FAILED $1 (see $OUT)"; return 1; }
}

run_pytest() { # dir tag -- pytest args...
  local d=$1 tag=$2; shift 3
  ( cd "$d" && echo "== $tag $(git rev-parse --short=8 HEAD)" \
    && PYTHONPATH=.:$EXTRA RL_KERNEL_H3_WEIGHTS=$WEIGHTS $PY -c \
       "import rl_engine.kernels.ops.base as b; print('_C', b._C.__file__)" \
    && PYTHONPATH=.:$EXTRA RL_KERNEL_H3_WEIGHTS=$WEIGHTS $PY "$HERE/preimport_pytest.py" "$@" -q --tb=short -p no:cacheprovider -rsf ) \
    > "$OUT/$tag.log" 2>&1
  echo "$tag: $(grep -E 'passed|failed|error' "$OUT/$tag.log" | tail -1)"
}

# 539a-c: strict build (also builds the normal _C), real weights, 2 GPUs
for b in "${H3[@]}"; do
  d=$(wt "$b"); build "$d" 1 || continue
  ( export CUDA_VISIBLE_DEVICES=0,1; run_pytest "$d" "$(basename "$d")-h3" -- \
      tests/models/minimax_h3 tests/backends/cuda/test_h3_full_attention.py )
done
# 4/8-GPU NCCL byte-equality (tests live in 539a). WORLD_SIZE must be unset: they spawn their own
# processes. The 8-GPU parameters skip on a node with fewer than 8 GPUs.
d=$(wt "${H3[0]}")
( unset WORLD_SIZE RANK LOCAL_RANK CUDA_VISIBLE_DEVICES; run_pytest "$d" "539a-nccl-4-8gpu" -- \
    tests/models/minimax_h3/test_h3_sp_norm_adaln.py tests/models/minimax_h3/test_h3_tp_adaln.py -k nccl -v )

# 543a-c: normal build, Qwen3-Next + shared validation tests, plus the Hopper-only SM90 test
for b in "${QN[@]}"; do
  d=$(wt "$b"); build "$d" 0 || continue
  ( export CUDA_VISIBLE_DEVICES=0,1; run_pytest "$d" "$(basename "$d")-qn" -- \
      tests/models/qwen3_next tests/validation/common )
  ( export CUDA_VISIBLE_DEVICES=0; run_pytest "$d" "$(basename "$d")-sm90" -- \
      tests/validation/common/test_logprob_comparison.py -k sm90_diagnostic -v )
done

echo; echo "summary ($OUT):"; grep -H -E "passed|failed|error" "$OUT"/*.log | grep -v build_ | sed 's#.*/##' | cut -c1-140
