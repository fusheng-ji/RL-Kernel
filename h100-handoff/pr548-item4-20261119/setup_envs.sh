#!/usr/bin/env bash
set -euxo pipefail
R=${R:?set R to the environment root}
export UV_CACHE_DIR=$R/uv-cache
IDX=https://download.pytorch.org/whl/cu130
# trainer: Py3.11, torch 2.9.1+cu130, transformers 5.13.1
uv venv -p 3.11 $R/envs/trainer
uv pip install -p $R/envs/trainer/bin/python --index-url $IDX torch==2.9.1
uv pip install -p $R/envs/trainer/bin/python pip "transformers==5.13.1" accelerate tabulate numpy==1.26.4 safetensors pytest ninja setuptools wheel
# framework overlay: megatron-core + deepspeed, separate target dir
$R/envs/trainer/bin/python -m pip install --no-deps --target $R/envs/frameworks \
  deepspeed==0.19.7 megatron-core==0.15.3 einops==0.8.2 hjson==3.1.0 msgpack==1.2.3 py-cpuinfo==9.0.0
# vLLM: separate interpreter
uv venv -p 3.11 $R/envs/vllm
uv pip install -p $R/envs/vllm/bin/python --extra-index-url $IDX --index-strategy unsafe-best-match vllm==0.15.1 pip
for p in trainer vllm; do $R/envs/$p/bin/python -c "import torch,sys; print(sys.version.split()[0], torch.__version__, torch.version.cuda)"; done
$R/envs/vllm/bin/python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"
PYTHONPATH=$R/envs/frameworks $R/envs/trainer/bin/python -c "import megatron.core, deepspeed; print('ok')" || true
