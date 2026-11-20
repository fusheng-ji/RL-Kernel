# header.sh <worktree-name> ; run on the H100 node
source <WORK>/pr536/env.sh
n=$1; cd $P/$n; mkdir -p $OUT/$n
{
echo "worktree: $n"; echo "sha: $(git rev-parse HEAD)"; echo "git status (tracked):"; git status --porcelain --untracked-files=no; echo "(end status)"
echo "diff summary:"; git diff --summary | awk '{print $1,$2,$3,$4}' | sort | uniq -c
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv
nvidia-smi | grep -i "CUDA Version" | sed 's/.*\(CUDA Version: [0-9.]*\).*/\1/'
echo "other compute processes on node:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | wc -l
PYTHONPATH=. $PY -c "import sys,torch,triton,rl_engine; print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__,triton.__file__)"
echo "PYTHONNOUSERSITE=$PYTHONNOUSERSITE RL_KERNEL_MODE=$RL_KERNEL_MODE"
} > $OUT/$n/header.txt 2>&1
