#!/bin/bash
# V1, CPU only (a node with Hub access; about 35 GiB download, 27 GiB result; +52 GiB with --bf16).
# DeepSeek-V4-Flash-Base at revision 8855555d: only the 7 shards holding the embedding, the head and
# layers 0-3, then the 4-layer slice (MTP dropped). Hashes are checked against ours in reference/.
set -euo pipefail
: "${H:?export H=<THIS_DIR>}"; source $H/env.sh
REV=8855555deef230a27a21a8d6f294b7b7497759b6
PY=${PY:-python3}   # needs safetensors and torch (CPU is enough); the hf CLI comes with huggingface_hub
SRC=$W/ckpt/full-partial; mkdir -p $W/ckpt
FILES="config.json tokenizer.json tokenizer_config.json model.safetensors.index.json $(for i in 1 2 3 4 5 45 46; do printf 'model-%05d-of-00046.safetensors ' $i; done)"
${HF:-hf} download deepseek-ai/DeepSeek-V4-Flash-Base $FILES --revision $REV --local-dir $SRC   # HF=huggingface-cli for older hubs
(cd $SRC && sha256sum -c $H/reference/ckpt-download.sha256)
[ -d $W/ckpt/v4-4l ] || $PY $H/slice_ckpt.py $SRC $W/ckpt/v4-4l 4
(cd $W/ckpt/v4-4l && sha256sum -c $H/reference/ckpt-v4-4l.sha256) || echo "SLICE_HASH_MISMATCH (report it; the run may still go on)"
if [ "${1:-}" = --bf16 ]; then
  [ -d $W/ckpt/v4-4l-bf16 ] || $PY $H/dequant_ckpt.py $W/ckpt/v4-4l $W/ckpt/v4-4l-bf16
  (cd $W/ckpt/v4-4l-bf16 && sha256sum -c $H/reference/ckpt-v4-4l-bf16.sha256) || echo "BF16_HASH_MISMATCH (report it)"
fi
