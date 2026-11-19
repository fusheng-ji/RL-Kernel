#!/usr/bin/env bash
# run_h100_full_chain.sh KIND OUT [ARGS...]   KIND: env | tests | run | compare
# phase_b_full_chain on one H100 (cc 9.0) from a clean checkout of feat/h3-phase-b-full-chain.
# Set SRC, PY, CKPT, AUDIO_VAE, PROVIDER and TEXT first (see HANDOFF.md). Nothing in SRC is edited.
set -euo pipefail
kind=$1 out=$2; shift 2
: "${SRC:?worktree of feat/h3-phase-b-full-chain}"
: "${PY:?python of the torch 2.13.0+cu130 environment}"
: "${CKPT:?dir holding transformer/ transformer_ref/ vae/ (MiniMax-H3@42ed227)}"
: "${AUDIO_VAE:?Hub-layout FL2VA/audio_vae dir (config.json, metadata.json, model.safetensors)}"
: "${PROVIDER:?diffusers checkout at f53d552036a0d1bd5570782a39cd40cfabf112bc}"
: "${TEXT:?dir holding fl2va-text.safetensors and ref2va-text.safetensors from this handoff}"
cd "$SRC"
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "dirty tree" >&2; exit 1; }
[[ $(git -C "$PROVIDER" rev-parse HEAD) == f53d552036a0d1bd5570782a39cd40cfabf112bc ]]
mkdir -p "$out"
export PYTHONPATH="$SRC:$PROVIDER/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$out/tmp"; mkdir -p "$TMPDIR"
git rev-parse HEAD > "$out/commit.txt"
T="tools/validation/models/h3_full_chain.py"
case $kind in
env)
  { nvidia-smi; "$PY" - <<'PY'
import torch
from rl_engine.backends.extension import _C
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0)); print("_C", _C.__file__)
from rl_engine.validation.models.h3_block import CandidateOps
ops = CandidateOps(); print("candidate backends", ops.backends())
print("h3_full_attention", ops.attention_h3 is not None)
assert torch.cuda.get_device_capability(0) == (9, 0), "this handoff is for Hopper (sm90)"
PY
  } > "$out/environment.txt" 2>&1 ;;
tests)
  "$PY" -m pytest tests/models/minimax_h3/test_h3_full_chain.py -q -rs -p no:cacheprovider \
    --junitxml="$out/tests.xml" > "$out/tests.txt" 2>&1 || true ;;
run)      # run WORKFLOW LABEL [extra run args]
  wf=$1 label=$2; shift 2
  td=$CKPT/transformer; [[ $wf == ref2va ]] && td=$CKPT/transformer_ref
  rc=0; "$PY" -W ignore "$T" run --workflow "$wf" --transformer-dir "$td" --vae-dir "$CKPT/vae" \
    --audio-vae-dir "$AUDIO_VAE" --text "$TEXT/$wf-text.safetensors" --text-sha256 "$(cat "$TEXT/$wf-text.sha256")" \
    --label "$label" --out "$out/$wf-$label.json" --report "$out/$wf-$label-report.json" "$@" \
    > "$out/$wf-$label.log" 2>&1 || rc=$?; echo "$wf $label exit=$rc" >> "$out/status.txt" ;;
compare)  # compare NAME LHS.json RHS.json [--tensors-lhs A --tensors-rhs B]
  name=$1; shift
  rc=0; "$PY" -W ignore "$T" compare "$@" --report "$out/cmp-$name.json" > "$out/cmp-$name.log" 2>&1 || rc=$?
  echo "cmp $name exit=$rc" >> "$out/status.txt" ;;
esac
