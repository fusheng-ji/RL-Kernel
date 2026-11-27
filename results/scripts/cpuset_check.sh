#!/usr/bin/env bash
# Inside job driver after slurm_job_env_exports: verify nested srun sees full node CPU sets (64 per node).
D=$(dirname "$0"); source "$D/common.sh"
J=${SLURM_JOB_ID:?}
O=$OUT/checks/miles-b3-cpuset.txt
mkdir -p "$OUT/checks"
{
  echo "job=$J time=$(date -u +%FT%TZ)"
  srun --jobid="$J" -N2 --ntasks-per-node=1 --overlap bash -c \
    'echo $(hostname -s | md5sum | cut -c1-6) $(grep Cpus_allowed_list /proc/self/status) nproc=$(nproc)'
} >"$O" 2>&1
lines=$(grep -c 'nproc=' "$O" || true)
ok=0
grep -q 'nproc=64' "$O" && [ "$lines" -ge 2 ] && ok=1
if [ "$ok" != 1 ]; then
  log "STOP: cpuset check failed — see checks/miles-b3-cpuset.txt"
  exit 1
fi
log "cpuset check OK ($lines nodes with nproc=64)"