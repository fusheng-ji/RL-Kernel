# The only file to fill in. Every job script sources it; submit with `export H=<THIS_DIR>` set
# (sbatch passes the environment through), see README "准备".
#   W        shared work directory (src/, ckpt/, enroot/, runs/, jitcache/ are created under it)
#   SCRATCH  node-local disk; each job uses $SCRATCH/vime-dsv4-$SLURM_JOB_ID and removes it at the end
export W=${W:-<WORK>}
export SCRATCH=${SCRATCH:-<LOCAL_SCRATCH>}
case "$W$SCRATCH" in *'<'*) echo "env.sh: set W and SCRATCH first" >&2; exit 2;; esac
export W=$(realpath "$W") H=$(realpath "$H")
export ENV_SQSH=${ENV_SQSH:-$W/enroot/vime-v4-env-sm90.sqsh}   # V0's output
