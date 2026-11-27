#!/bin/bash
# h0_prepare.sh minus the enroot import (no enroot on the login node); same pins.
set -uo pipefail
WORK=$SCRATCH_ROOT/sched_20261127/miles
export HF_HOME=$WORK/cache/hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 TMPDIR=$WORK/tmp
unset HF_TOKEN WANDB_API_KEY
HF=~/miniconda3/bin/hf
MODEL_REV=7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6
DS_REV=2e65612930298bde4c5d58fd97b3f23a483aaff9
TOKEN=$(curl -sf "https://auth.docker.io/token?service=registry.docker.io&scope=repository:radixark/miles:pull" | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -sf -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json" https://registry-1.docker.io/v2/radixark/miles/manifests/inkling-2-mini > $WORK/runs/h0/manifest_index.json
python3 - $WORK/runs/h0/manifest_index.json <<'PY' | tee $WORK/runs/h0/image_digest.txt
import json, sys
m = json.load(open(sys.argv[1]))
for x in m.get("manifests", []):
    if x.get("platform", {}).get("architecture") == "amd64": print("amd64", x["digest"])
PY
$HF download thinkingmachines/Inkling-2-Mini --revision $MODEL_REV --local-dir $WORK/models/Inkling-2-Mini || echo "MODEL_DL_FAIL"
$HF download --repo-type dataset zhuzilin/dapo-math-17k --revision $DS_REV --local-dir $WORK/datasets/dapo-math-17k || echo "DS_DL_FAIL"
printf 'model %s\ndataset %s\n' $MODEL_REV $DS_REV > $WORK/runs/h0/revisions.txt
cd $WORK && find models/Inkling-2-Mini datasets/dapo-math-17k -type f -not -path '*/.cache/*' -print0 | sort -z | xargs -0 sha256sum > runs/h0/download.sha256
echo MILES_DL_DONE
