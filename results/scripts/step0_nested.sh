#!/usr/bin/env bash
# B1/B2 gate: nested srun must see two host hashes and 4 GPUs each (miles-style and vime-style).
source "$(dirname "$0")/common.sh"
D=$(dirname "$0")
J=$1; NA=$2; NB=$3
O=$OUT/checks/nested-srun-2node.txt
DRIVER_MODE=${DRIVER_MODE:-inside}   # inside | login
PROBE='echo $(hostname -s | md5sum | cut -c1-6) $(nvidia-smi -L 2>/dev/null | wc -l) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}'
nested_run() {
  local mode=$1
  : > "$O"
  echo "DRIVER_MODE=$mode SLURM_JOB_ID=$J" >> "$O"
  if [ "$mode" = login ]; then
    (
      slurm_job_env_exports "$J"
      echo "=== miles: srun --ntasks-per-node=1 --overlap ===" >> "$O"
      srun --jobid="$J" --ntasks-per-node=1 --overlap bash -c "$PROBE" >> "$O" 2>&1 || echo "miles_rc=$?" >> "$O"
      echo "=== vime: srun -N\$SLURM_NNODES --ntasks-per-node=1 --overlap ===" >> "$O"
      srun --jobid="$J" -N"$SLURM_NNODES" --ntasks-per-node=1 --overlap bash -c "$PROBE" >> "$O" 2>&1 || echo "vime_rc=$?" >> "$O"
    )
  else
    srun --jobid="$J" -w "$NA" --overlap -N1 -n1 -c 2 --gres=none --mem=0 \
      /usr/bin/env SLURM_GRES=gpu:4 SLURM_OVERLAP=1 bash -c "
        source $D/common.sh
        slurm_job_env_exports $J
        echo '=== miles: srun --ntasks-per-node=1 --overlap ===' 
        srun --ntasks-per-node=1 --overlap bash -c '$PROBE'
        echo '=== vime: srun -N'\$SLURM_NNODES' --ntasks-per-node=1 --overlap ==='
        srun -N\$SLURM_NNODES --ntasks-per-node=1 --overlap bash -c '$PROBE'
      " >> "$O" 2>&1
  fi
}
case "$DRIVER_MODE" in
  login|login_bash) nested_run login;;
  *) nested_run inside;;
esac
python3 - "$O" <<'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read()
# hash gpus cuda — lines like: abc123 4 CUDA_VISIBLE_DEVICES=0,1,2,3
pat = re.compile(r"^([0-9a-f]{6}) (\d+) CUDA_VISIBLE_DEVICES=(\S+)", re.M)
blocks = re.split(r"^=== ", t, flags=re.M)
def block_ok(name_part):
    for blk in blocks:
        if blk.startswith(name_part):
            ms = pat.findall(blk)
            if len(ms) < 2:
                return False, ms
            hashes = {m[0] for m in ms}
            if len(hashes) < 2:
                return False, ms
            if not all(int(m[1]) == 4 for m in ms):
                return False, ms
            return True, ms
    return False, []
m_ok, m_ms = block_ok("miles")
v_ok, v_ms = block_ok("vime")
ver = f"ok={m_ok and v_ok} miles={m_ok} vime={v_ok} miles_lines={len(m_ms)} vime_lines={len(v_ms)}\n"
open(sys.argv[1] + ".verdict", "w").write(ver)
if m_ms:
    open(sys.argv[1] + ".verdict", "a").write(f"miles_sample={m_ms[0]}\n")
sys.exit(0 if m_ok and v_ok else 1)
PY
rc=$?
if [ $rc = 0 ]; then gate_set nested_srun OK; else gate_set nested_srun "FAIL see $O"; fi
exit $rc