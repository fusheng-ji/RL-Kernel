#!/bin/bash
# H1 (one node): record the software, convert to BF16 twice (bitwise check), run the PR's own tests.
set -x
cd /root/miles
{ date -u; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nvidia-smi topo -m | head -12
  git -C /root/miles rev-parse HEAD; git -C /root/miles status --porcelain -uno | head
  git -C /sgl-workspace/sglang rev-parse HEAD; git -C /sgl-workspace/sglang status --porcelain -uno | head
  git -C /root/torchtitan rev-parse HEAD 2>/dev/null || true
  python3 -c "import torch,sys;print('torch',torch.__version__,torch.version.cuda,torch.cuda.device_count())"
  python3 -m pip list 2>/dev/null | grep -iE '^(torch|torchtitan|sglang|sglang-kernel|sglang-router|flashinfer|transformers|ray|tml-renderers|miles|flash-attn|triton|numpy|transformer)'
  env | grep -E 'PYTHONPATH|SGLANG|CUDA|LD_LIBRARY' | sort
} > $RUN/env.txt 2>&1
cat $RUN/env.txt
mkdir -p $WORK/tmp
S=$WORK/models/Inkling-2-Mini
for i in a b; do
  D=$WORK/models/Inkling-2-Mini-bf16; [ $i = b ] && D=$WORK/tmp/bf16-check
  [ $i = b ] && rm -rf $D
  [ -f $D/model.safetensors.index.json ] || python3 tools/convert_inkling2_to_bf16.py --model-dir $S --save-dir $D > $RUN/convert_$i.log 2>&1
  (cd $D && find . -type f -not -path './.cache/*' -print0 | sort -z | xargs -0 sha256sum) > $RUN/bf16_$i.sha256
done
if cmp -s $RUN/bf16_a.sha256 $RUN/bf16_b.sha256; then echo BF16_BYTE_IDENTICAL; else echo BF16_DIFFER; diff $RUN/bf16_a.sha256 $RUN/bf16_b.sha256 | head; fi
rm -rf $WORK/tmp/bf16-check
python3 - <<'PY' > $RUN/bf16_tensors.json
import json,hashlib,os
from safetensors import safe_open
d=os.environ["WORK"]+"/models/Inkling-2-Mini-bf16"
idx=json.load(open(d+"/model.safetensors.index.json"))["weight_map"]
out={}
for f in sorted(set(idx.values())):
    with safe_open(f"{d}/{f}","pt") as s:
        for k in s.keys():
            t=s.get_tensor(k).contiguous()
            out[k]=[str(t.dtype),list(t.shape),hashlib.sha256(t.flatten().view(__import__("torch").uint8).numpy().tobytes() if t.numel() else b"").hexdigest()]
json.dump(out,__import__("sys").stdout,indent=0)
PY
echo "bf16 tensors: $(python3 -c "import json;print(len(json.load(open('$RUN/bf16_tensors.json'))))")"
export CUDA_VISIBLE_DEVICES=0
for t in tests/fast/backends/test_torchtitan_inkling2.py tests/fast/backends/test_torchtitan_hf_export.py tests/fast/tools/test_convert_inkling2_to_bf16.py tests/fast-gpu/test_torchtitan_inkling2_forward.py; do
  timeout 1200 python3 -m pytest -q -p no:cacheprovider $t > $RUN/pytest_$(basename $t .py).log 2>&1; echo "$t rc=$?"; tail -4 $RUN/pytest_$(basename $t .py).log
done
