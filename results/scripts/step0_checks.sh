#!/usr/bin/env bash
# Step 0 per node: card, driver, IB, local disk, Hub, enroot, shared RW, other jobs on this node (count only).
source "$(dirname "$0")/common.sh"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-}")
H=$(hostname -s)
if [ "${NODES[0]:-}" = "$H" ]; then ROLE=node-a; else ROLE=node-b; fi
O=$OUT/checks/step0-$ROLE; mkdir -p $O
nvidia-smi > $O/nvidia-smi.txt 2>&1
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv > $O/gpus.csv
(ibstat || ibv_devinfo) > $O/ib.txt 2>&1
lscpu | grep -E 'Model name|^CPU\(s\)|NUMA' > $O/cpu.txt; nvidia-smi topo -m > $O/topo.txt 2>&1
df -h /tmp /dev/shm / > $O/df-local.txt 2>&1
echo "TMPDIR=${TMPDIR:-unset} SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}" >> $O/df-local.txt
(curl -sI -m 10 https://huggingface.co | head -1; curl -sI -m 10 https://registry-1.docker.io/v2/ | head -1) > $O/hub.txt 2>&1
(which enroot; enroot version; srun --help 2>/dev/null | grep -c container-image) > $O/enroot-pyxis.txt 2>&1
echo "other_jobs_on_node=$(squeue -h -w "$H" 2>/dev/null | wc -l)" >> $O/env-squeue.txt
echo ok-$ROLE > $SCHED/checks/rw-$ROLE.txt
cat $SCHED/checks/rw-$ROLE.txt > $O/shared-rw.txt
# VIME local enroot scratch probe (gate only lane 3 if too small)
VIME_LOCAL=${VIME_LOCAL:-/tmp}
df -BM "$VIME_LOCAL" 2>/dev/null | tail -1 > $O/vime-local-df.txt