#!/usr/bin/env bash
# Like areal-rlkg <BUNDLE_SCRIPTS_DIR>/run_verify.sh but cwd is OUT (HANDOFF: do not cd into RL-Kernel source tree).
set -uo pipefail
HEAD_IP="${1:?head ip}"; OUT="${2:?out dir}"; shift 2
BUNDLE="${AREAL_BUNDLE:-$HOME/datastorage/areal-rlkg-1121}"
HERE="$BUNDLE/<BUNDLE_SCRIPTS_DIR>"
for v in $(env | cut -d= -f1 | grep -E '^SLURM_' || true); do unset "$v"; done
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1
# shellcheck disable=SC1091
source "$HERE/cluster_env.sh"
# shellcheck disable=SC1091
source "$BUNDLE/activate.sh"
export RLK_AREAL_WORKERS_READY_TIMEOUT="${RLK_AREAL_WORKERS_READY_TIMEOUT:-300}"
export RAY_ADDRESS="$HEAD_IP:${RAY_PORT:-26379}"
ray status >/dev/null || { echo "ray cluster not reachable" >&2; exit 1; }
mkdir -p "$OUT"
cd "$OUT"
start=$SECONDS
python -m rl_engine.integrations.orchestrators.areal.verify run \
  --areal-root "$AREAL_ROOT" --model "$MODEL" --out "$OUT" \
  --scheduler ray --nodes 2 --gpus 4 --dataset-path "$GSM8K_PATH" \
  --mode sync --vllm-execution graph "$@"
code=$?
echo "verify run exit=$code seconds=$((SECONDS - start)) args=$*" | tee -a "$OUT/../verify-status.txt"
exit $code