#!/usr/bin/env bash
# H100 diagnosis of the rollout-smoke abort. Needs 4 GPUs, the rlkernel env (vllm 0.30.0,
# flashinfer-python 0.6.18.post1), ninja on PATH, and local copies of Qwen/Qwen3.5-0.8B and
# Qwen/Qwen2-1.5B-Instruct (config + tokenizer only: weights are dummy).
# Usage: run_probe.sh <python> <qwen3.5-0.8b dir> <qwen2-1.5b dir> <out dir>
set -uo pipefail
PY=$1; Q35=$2; Q2=$3; O=$4; HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$O"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export TMPDIR="/tmp/fi-$$"; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT   # short: vLLM IPC path limit
export FLASHINFER_WORKSPACE_BASE="$O/fi-cache"; mkdir -p "$FLASHINFER_WORKSPACE_BASE"  # cold JIT cache
{
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv
  nvidia-smi topo -m
  nvidia-smi -q | grep -iE "fabric|multicast|nvlink" | sort | uniq -c
  "$PY" -c 'import torch,vllm,flashinfer,tvm_ffi,sys;print("python",sys.version.split()[0],"torch",torch.__version__,"vllm",vllm.__version__,"flashinfer",flashinfer.__version__,"tvm_ffi",tvm_ffi.__version__, tvm_ffi.__file__)'
  "$PY" -m pip list 2>/dev/null | grep -iE "flashinfer|tvm|nvidia-nccl|cuda-python"
  # two copies of the tvm-ffi runtime would explain a type attribute registered twice
  "$PY" - <<'EOF'
import site, glob, os
for d in site.getsitepackages():
    for f in glob.glob(os.path.join(d, "**", "libtvm_ffi*.so*"), recursive=True):
        print("libtvm_ffi:", f.replace(d, "<site-packages>"))
EOF
  echo "default FlashInfer JIT cache entries: $(ls ~/.cache/flashinfer 2>/dev/null | tr '\n' ' ')"
} > "$O/header.txt" 2>&1
case_run() {  # case_run <name> <model> <env...> -- [probe flags]
  local n=$1 m=$2; shift 2; local envs=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done; [ "${1:-}" = -- ] && shift
  echo "[$(date -u +%FT%TZ)] START $n ${envs[*]} $*" >> "$O/steps.txt"
  env "${envs[@]}" timeout 1200 "$PY" "$HERE/probe.py" --model "$m" --out "$O/$n.json" "$@" > "$O/$n.log" 2>&1
  local rc=$?
  echo "[$(date -u +%FT%TZ)] END $n rc=$rc" >> "$O/steps.txt"
  printf '%-26s rc=%-4s ffi_abort=%s workspace=%s\n' "$n" "$rc" \
    "$(grep -c '__ffi_repr__' "$O/$n.log")" \
    "$(grep -o 'norm fusion workspace with backend=[a-z]*' "$O/$n.log" | head -1)" >> "$O/summary.txt"
}
case_run q35-auto          "$Q35" VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto --
case_run q35-auto-nofuse   "$Q35" VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto -- --no-fuse
case_run q35-no-fi-ar      "$Q35" VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto VLLM_ALLREDUCE_USE_FLASHINFER=0 -- --no-fuse
case_run q35-gdntriton     "$Q35" VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto -- --gdn-triton
case_run q2-auto           "$Q2"  VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto --
case_run q2-no-fi-ar       "$Q2"  VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto VLLM_ALLREDUCE_USE_FLASHINFER=0 -- --no-fuse
touch "$O/ALLDONE"
