#!/bin/bash
#SBATCH --job-name=v4-build
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
# V0: import the pinned vLLM image, build the remaining packages for sm90 inside it, export the result.
# Built wheels are kept in $W/wheels-sm90, so a resubmission after a failure or preemption skips them.
set -uo pipefail
: "${H:?export H=<THIS_DIR> before sbatch}"; source $H/env.sh
R=$SCRATCH/vime-dsv4-$SLURM_JOB_ID; mkdir -p $R/{data,run,tmp} $W/enroot $W/logs; trap 'rm -rf $R' EXIT
export ENROOT_CACHE_PATH=$W/enroot/cache ENROOT_DATA_PATH=$R/data ENROOT_RUNTIME_PATH=$R/run ENROOT_TEMP_PATH=$R/tmp TMPDIR=$R/tmp ENROOT_SQUASH_OPTIONS="-comp zstd -noD"
IMG=$W/enroot/vllm-cb12c30.sqsh
if [ ! -f $IMG ]; then
  # "@sha256:" would be parsed by enroot as user@host; ":sha256:" is the digest form it accepts
  enroot import -a x86_64 -o $R/vllm.sqsh "docker://vllm/vllm-openai:sha256:cb12c3016938a4ee98ed176ac833984089460689a0982cff4b5a60bb45b3cded" || exit 1
  cp $R/vllm.sqsh $W/enroot/.part.sqsh && mv $W/enroot/.part.sqsh $IMG
fi
sha256sum $IMG > $W/logs/image.sha256
enroot create --name vb $IMG
# the image's entrypoint swallows the command, hence --rc rc.sh (exec "$@")
enroot start --root --rw --rc $H/rc.sh -m $W:$W -m $H:$H -m $R:$R vb bash $H/build_inner.sh $W $R
test -f $R/ok || { echo BUILD_FAILED; exit 1; }
enroot export --output $W/enroot/vime-v4-env-sm90.sqsh --force vb; ls -la $W/enroot/*.sqsh
