#!/usr/bin/env bash
# run_h100.sh KIND SOURCE_DIR OUT_DIR [ARGS...]   KIND: build | tests | mypy | fulltrans | dual | ws2full
# FAST_MATH=1 builds and tests the KERNEL_ALIGN_USE_FAST_MATH=1 variant (use its own worktree: the build is in place).
# Runs one item of this handoff from a clean checkout on an H100 (cc 9.0). Set the site paths below (or in the
# environment) first. Stage-resumable: a finished stage leaves <stage>.done in OUT_DIR and is skipped on a rerun.
# Nothing in SOURCE_DIR is edited; bounds and runners are the committed ones.
set -euo pipefail
kind=$1 source_dir=$2 out=$3; shift 3
: "${PY:?python of the torch 2.13.0+cu130 environment}"
: "${CKPT:?dir holding transformer/ (MiniMax-H3@42ed227, 14 shards)}"
: "${WEIGHTS:?RL_KERNEL_H3_WEIGHTS dir (h3_conditioning, h3_block0, hf/transformer)}"
: "${PROVIDER:?diffusers checkout at f53d552036a0d1bd5570782a39cd40cfabf112bc}"
OVERLAY=${OVERLAY:-}            # extra site-packages for the provider's dependencies, if any
VLLM_OMNI=${VLLM_OMNI:-}        # vLLM-Omni checkout at 12e92800 (optional)
VERL_OMNI=${VERL_OMNI:-}        # verl-omni checkout at efb9dc7f (optional)

cd "$source_dir"
[[ -z $(git status --porcelain) ]] || { echo "dirty tree" >&2; exit 1; }
[[ $(git -C "$PROVIDER" rev-parse HEAD) == f53d552036a0d1bd5570782a39cd40cfabf112bc ]]
# ws2full: every rank writes under its own directory and never builds (build once first with KIND=build)
if [[ $kind == ws2full ]]; then out="$out/${WS2_NAME:?grid name, e.g. tp4-cp2-ring}/rank${SLURM_PROCID:?launch with srun}"; fi
mkdir -p "$out"; touch "$out/status.txt"
export PYTHONPATH="$source_dir:$PROVIDER/src${OVERLAY:+:$OVERLAY}${VLLM_OMNI:+:$VLLM_OMNI}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false KERNEL_ALIGN_USE_FAST_MATH=${FAST_MATH:-0} RL_KERNEL_H3_STRICT=1
export RL_KERNEL_H3_WEIGHTS="$WEIGHTS" ${VERL_OMNI:+VERL_OMNI_PATH="$VERL_OMNI"}
export TMPDIR="$out/tmp" TRITON_CACHE_DIR="$out/triton-cache" TORCH_KERNEL_CACHE_PATH="$out/torch-cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_KERNEL_CACHE_PATH"

git rev-parse HEAD >"$out/commit.txt"
"$PY" -m pip freeze >"$out/pip-freeze.txt"
if [[ $kind != ws2full && ! -f "$out/build.done" ]]; then
  "$PY" setup.py build_ext --inplace >"$out/build.log" 2>&1   # ninja on PATH makes this ~1 min
  touch "$out/build.done"
fi
[[ -z $(git status --porcelain) ]]
{ nvidia-smi; git -C "$PROVIDER" rev-parse HEAD
  for dep in "$VLLM_OMNI" "$VERL_OMNI"; do
    if [[ -n $dep ]]; then echo "$(basename "$dep") $(git -C "$dep" rev-parse HEAD)"; fi
  done
  "$PY" - <<'PY'
import importlib, torch, triton
from rl_engine.backends.extension import _C
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("capability", torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))
print("_C", _C.__file__)
try:
    print("_H3_C", importlib.import_module("rl_engine._H3_C").__file__)
except ImportError as exc:  # the fast-math build leaves the H3 kernels out
    print("_H3_C not built:", exc)
import os; print("KERNEL_ALIGN_USE_FAST_MATH", os.environ["KERNEL_ALIGN_USE_FAST_MATH"])
assert torch.cuda.get_device_capability(0) == (9, 0), "this handoff is for Hopper (sm90)"
PY
} >"$out/environment.txt" 2>&1

stage() {  # stage NAME CMD...
  local name=$1; shift
  [[ -f "$out/$name.done" ]] && { echo "skip $name"; return 0; }
  local rc=0; "$@" >"$out/$name.log" 2>&1 || rc=$?
  echo "$name exit=$rc" >>"$out/status.txt"
  [[ $rc == 0 ]] && touch "$out/$name.done"
  return 0   # keep going; failures are recorded in status.txt
}
case $kind in
tests)
  stage tests "$PY" -m pytest tests/models/minimax_h3 -q -rs -p no:cacheprovider --junitxml="$out/tests.xml" ;;
mypy)        # as CI runs it; with torch installed, 36 pre-existing errors in rl_engine/integrations are expected
  stage mypy "$PY" -m mypy --ignore-missing-imports rl_engine/ ;;
fulltrans)   # h3_full_transformer.py has no seed: one strict-vs-provider run, plus the provider-boundaries variant
  stage full-transformer "$PY" tools/validation/models/h3_full_transformer.py \
    --checkpoint "$CKPT" --output "$out/full-transformer"
  stage full-transformer-provider-boundaries "$PY" tools/validation/models/h3_full_transformer.py \
    --checkpoint "$CKPT" --output "$out/full-transformer-provider-boundaries" --provider-boundaries ;;
dual)
  for seed in 11 22 33; do
    stage "dual-trajectory-seed$seed" "$PY" tools/validation/models/h3_dual_trajectory.py \
      --checkpoint "$CKPT" --seed "$seed" --output "$out/dual-trajectory-seed$seed"
  done ;;
ws2full)     # run inside an srun allocation (one task per GPU); ARGS go to h3_parallel_full_model.py
  : "${MASTER_ADDR:?address of the first node}"
  stage ws2 "$PY" tools/validation/models/h3_parallel_full_model.py \
    --checkpoint "$CKPT" --local-size "${LOCAL_SIZE:-4}" --out "$out/../report" "$@" ;;
build) ;;   # only the build and environment steps above
*) echo "unknown kind $kind" >&2; exit 2 ;;
esac
cat "$out/status.txt"
! grep -q 'exit=[1-9]' "$out/status.txt"
