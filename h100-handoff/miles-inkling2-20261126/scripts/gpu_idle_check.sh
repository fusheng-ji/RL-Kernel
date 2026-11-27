#!/bin/bash
# Stage-start check: exactly 4 GPUs visible and none holds more than 1 GiB. Writes <out prefix>.txt (topology, NVLink,
# fabric state, per-GPU memory and processes, the CPUs this step may use) and prints "gpu_check ok" or "gpu_check busy: ...". Exit 0 = ok, 96 = busy.
# args: <out prefix>
OUT=$1
{
  echo "== memory"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  echo "== compute apps"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
  echo "== topo"; nvidia-smi topo -m
  echo "== nvlink"; nvidia-smi nvlink -s
  echo "== fabric"; nvidia-smi -q | grep -iA3 fabric
  echo "== cpus"; echo "nproc $(nproc)"; grep -E "^(Cpus|Mems)_allowed_list" /proc/self/status
  env | grep -E "^SLURM_(CPUS_PER_TASK|TRES_PER_TASK|CPUS_PER_GPU|MEM_PER_[A-Z]+|CPUS_ON_NODE|JOB_CPUS_PER_NODE)=" | sort
} > "$OUT.txt" 2>&1
n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2 > 1024 {printf "gpu%s=%sMiB ", $1, $2}')
if [ "$n" != 4 ] || [ -n "$busy" ]; then
  echo "gpu_check busy: visible=$n $busy" | tee -a "$OUT.txt"; exit 96
fi
echo "gpu_check ok" | tee -a "$OUT.txt"
