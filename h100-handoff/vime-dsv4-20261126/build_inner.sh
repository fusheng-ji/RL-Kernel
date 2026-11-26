#!/bin/bash
# Runs inside the container as (fake) root. Follows docker/Dockerfile with the PR's validated pins; base-image packages are frozen.
# H100 copy of the B200 build: only the CUDA arch lists, the TE/FlashMLA commits (pinned instead of tag/branch) and the wheel directory differ.
set -uxo pipefail
export FLASH_ATTN_CUDA_ARCHS=90
W=$1; WH=$W/wheels-sm90; mkdir -p $WH; cd /root; export PIP_NO_CACHE_DIR=1 TMPDIR=/root/tmp PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1; mkdir -p /root/tmp
wh(){ n=$1; shift; f=$(ls $WH/$n-*.whl 2>/dev/null | head -1); if [ -z "$f" ]; then d=$(mktemp -d /root/tmp/w.XXXX); python3 -m pip wheel --no-build-isolation --no-deps -w $d "$@" && f=$(ls $d/*.whl | head -1) && cp "$f" $WH/.part-$n && mv $WH/.part-$n "$WH/$(basename $f)"; f=$(ls $WH/$n-*.whl 2>/dev/null | head -1); fi; [ -n "$f" ] && python3 -m pip install -q --no-deps "$f" || echo STEP_FAIL $n; }
PY=python3; P="python3 -m pip install -q"; NB="python3 -m pip install --no-build-isolation --no-deps"
ln -sf "$(command -v python3)" /usr/local/bin/python
python3 -m pip list --format=freeze 2>/dev/null | grep -vE "^(pip|setuptools|wheel)=" > /root/base.txt; wc -l /root/base.txt
apt-get update -qq && apt-get install -y -qq git cmake cuda-nvrtc-dev-13-0 cuda-nvml-dev-13-0 cuda-profiler-api-13-0 cuda-nvtx-13-0 \
  libcusparse-dev-13-0 libcusolver-dev-13-0 libcufft-dev-13-0 libcurand-dev-13-0 libcublas-dev-13-0 libcudnn9-dev-cuda-13 libibverbs1 ibverbs-utils librdmacm1 || echo STEP_FAIL apt
$P setuptools wheel ninja pybind11 packaging || echo STEP_FAIL basics
export CPATH=/usr/local/cuda/include/cccl NVCC_PREPEND_FLAGS="-I/usr/local/cuda/include/cccl" NVCC_APPEND_FLAGS="--threads 4" TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32
for r in $(sed 's/#.*//' $W/src/vime/requirements.txt) ; do $P -c /root/base.txt "$r" || { echo "STEP_FAIL_CONSTRAINED $r"; $P "$r" || echo STEP_FAIL req $r; }; done
curl -sSL -o /root/fa.tgz https://files.pythonhosted.org/packages/source/f/flash_attn/flash_attn-2.8.3.tar.gz && mkdir -p /root/fa && tar xzf /root/fa.tgz -C /root/fa --strip-components=1 && \
  sed -i 's/c++17/c++20/g' /root/fa/setup.py && wh flash_attn /root/fa
wh fast_hadamard_transform git+https://github.com/Dao-AILab/fast-hadamard-transform.git@f134af63deb2df17e1171a9ec1ea4a7d8604d5ca
wh apex --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
NC=$(python3 -c 'import importlib.util;print(next(iter(importlib.util.find_spec("nvidia.nccl").submodule_search_locations)))')
mkdir -p /root/link; ln -sf $NC/lib/libnccl.so.2 /root/link/libnccl.so; CD=$(python3 -c 'import importlib.util;print(next(iter(importlib.util.find_spec("nvidia.cudnn").submodule_search_locations)))' 2>/dev/null); [ -n "$CD" ] && ln -sf $CD/lib/libcudnn.so.9 /root/link/libcudnn.so
ln -sf $NC/lib/libnccl.so.2 /usr/local/lib/libnccl.so; export CMAKE_LIBRARY_PATH=/root/link
CPATH=$NC/include:$CPATH LIBRARY_PATH=/root/link LDFLAGS=-L/root/link NVTE_CUDA_ARCHS="90" wh transformer_engine git+https://github.com/NVIDIA/TransformerEngine.git@27486e03cfc1fa41f6932dcecdc47c71c47eac3e
rm -f /usr/local/lib/libnccl.so
git clone -q -b nv_dev https://github.com/deepseek-ai/FlashMLA /root/FlashMLA && (cd /root/FlashMLA && git submodule update -q --init --recursive && git checkout -q b7643bd54521f563b839b98289b5cd048c062ba2 && git submodule update -q --init --recursive && git rev-parse HEAD && wh flash_mla .) || echo STEP_FAIL flashmla
TMS_CUDA_MAJOR=13 $P --no-deps git+https://github.com/zhuzilin/torch_memory_saver.git@8d30c59ca12a68d9deccbc9c6599076a1218cbc5 || echo STEP_FAIL tms
for r in onnxscript nvdlfw-inspect nvidia-resiliency-ext einops pytest; do $P -c /root/base.txt $r || echo STEP_FAIL req $r; done
python3 -m pip list --format=freeze 2>/dev/null > $WH/env-freeze-sm90.txt; diff <(sort /root/base.txt) <(sort $WH/env-freeze-sm90.txt | grep -vE "^(pip|setuptools|wheel)=") | grep '^<' | head
rm -rf /root/tmp /root/FlashMLA/build /root/fa /root/fa.tgz; python3 -c "import torch;assert torch.cuda.get_device_capability()==(9,0),torch.cuda.get_device_capability();import vllm,transformer_engine,transformer_engine.pytorch,apex,flash_attn,fast_hadamard_transform,flash_mla,torch_memory_saver;print('IMPORTS_OK')" && touch $2/ok; echo BUILD_DONE
