# Shared helpers for the B200 gap-filling runs. Source from a per-PR run script.
# Required before sourcing: GAPS (root of the worktrees), WT (worktree under test), OUT (result directory).
PY=$HOME/miniconda3/envs/rlkernel/bin/python
TORCHRUN=$HOME/miniconda3/envs/rlkernel/bin/torchrun
export PATH=$GAPS/ninja-bin:$HOME/miniconda3/envs/rlkernel/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=$WT
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/tmp"
export TMPDIR=${TMPDIR:-$OUT/tmp}
cd "$WT"

env_header() {
  {
    echo "date_utc: $(date -u +%FT%TZ)"
    echo "head_sha: $(git -C "$WT" rev-parse HEAD)"
    echo "git_status_short: [$(git -C "$WT" status --short)]"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    nvcc --version | tail -2
    $PY - <<'PYEOF'
import sys, torch, triton
import rl_engine
from rl_engine.kernels.ops import base
print("python", sys.version.split()[0], "torch", torch.__version__, "triton", triton.__version__)
print("rl_engine", rl_engine.__file__)
print("_C", base._C.__file__ if getattr(base, "_C", None) else None)
PYEOF
  } > "$OUT/environment.txt" 2>&1
  cat "$OUT/environment.txt"
  grep -q "^_C $WT/" "$OUT/environment.txt" || { echo "FATAL: _C does not resolve into the worktree under test"; exit 3; }
}

# run NAME CMD...: log to $OUT/NAME.log, record command and exit code, never abort the job.
run() {
  local name=$1; shift
  echo "### $name: $*"
  printf '%s\t%s\n' "$name" "$*" >> "$OUT/commands.tsv"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  printf '%s\t%s\n' "$name" "$rc" >> "$OUT/exitcodes.tsv"
  echo "### $name exit $rc"
  tail -3 "$OUT/$name.log"
}

finish() {
  echo "git_status_short_after_run: [$(git -C "$WT" status --short)]" >> "$OUT/environment.txt"
  echo "head_sha_after_run: $(git -C "$WT" rev-parse HEAD)" >> "$OUT/environment.txt"
}
