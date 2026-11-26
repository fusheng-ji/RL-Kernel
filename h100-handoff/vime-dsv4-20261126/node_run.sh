#!/bin/bash
# One per node, inside the container. Args: ROLE(head|worker) HEAD_IP NGPU MY_IP IFNAME
# Env (from common.sh): W H RUN TOPO STEPS LR EXTRA EXTRA_ENV OPT_ARGS NNODES TOPO_ARGS CKPT VIMEDIR
set -x
: "${OPT_ARGS:=--use-stateless-adam --use-precision-aware-optimizer --no-save-optim --reset-optimizer-states}"
[ -n "${EXTRA_ENV:-}" ] && export $EXTRA_ENV
ROLE=$1; HEAD=$2; NGPU=$3; MYIP=$4; IFN=$5
export PYTHONNOUSERSITE=1 HOME=/tmp/home PYTHONUNBUFFERED=1 V4_ALLOW_NUMPY2=1; mkdir -p $HOME $RUN
# patch 0003: upstream vLLM 14a5bb4ad backport (V4 post-load hook during weight sync), applied to the container copy
(cd /usr/local/lib/python3.12/dist-packages && patch -p1 --forward < $H/patches/0003-vllm-backport-postload-hook.patch) > $RUN/vllm-patch-$(hostname).log 2>&1 || { echo VLLM_PATCH_FAILED; cat $RUN/vllm-patch-$(hostname).log; }
echo numpy-mode=numpy2-with-patch0002 > $RUN/env.txt
export XDG_CACHE_HOME=$W/jitcache VLLM_CACHE_ROOT=$W/jitcache/vllm TILELANG_CACHE_DIR=$W/jitcache/tilelang FLASHINFER_WORKSPACE_BASE=$W/jitcache TRITON_CACHE_DIR=/tmp/triton-cache TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
export PYTHONPATH=$VIMEDIR:$W/src/Megatron-run:$H/probe V4PROBE_DIR=$RUN
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_DEBUG_FILE=$RUN/nccl-%h-%p.log CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_CUMEM_ENABLE=0
# use only GPUs Slurm gave this job (the first NGPU of them)
: "${CUDA_VISIBLE_DEVICES:?no GPUs from Slurm}"; export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | cut -d, -f1-$NGPU)
echo CVD=$CUDA_VISIBLE_DEVICES; nvidia-smi -L | head -8
# each case gets its own Ray port, so a case that has not finished shutting down cannot collide
RAY_PORT=$((20000 + $(echo -n "$RUN" | cksum | cut -d" " -f1) % 9000)); export RAY_ADDRESS=$HEAD:$RAY_PORT
export NCCL_SOCKET_IFNAME=$IFN GLOO_SOCKET_IFNAME=$IFN TP_SOCKET_IFNAME=$IFN; echo IFN=$IFN MYIP=$MYIP RAY_ADDRESS=$RAY_ADDRESS
if [ "$ROLE" = worker ]; then
  for i in $(seq 60); do ray start --address=$HEAD:$RAY_PORT --num-gpus $NGPU --node-ip-address $MYIP --disable-usage-stats && break; sleep 5; done
  while [ ! -f $RUN/DONE ]; do sleep 5; done
  exit 0
fi
ray start --head --port $RAY_PORT --ray-client-server-port $((RAY_PORT+9500)) --node-ip-address $HEAD --num-gpus $NGPU --disable-usage-stats --include-dashboard=false
for i in $(seq 60); do n=$(python3 -c "import ray;ray.init(address='auto',log_to_driver=False);print(len([x for x in ray.nodes() if x['Alive']]))" 2>/dev/null | tail -1); [ "$n" = "${NNODES:-1}" ] && break; sleep 5; done
cd $VIMEDIR
source scripts/models/deepseek-v4-flash.sh
python3 train.py \
  "${MODEL_ARGS[@]}" --hf-checkpoint $CKPT --load $CKPT --ref-load $CKPT \
  --prompt-data $H/data/gsm8k.jsonl --input-key prompt --label-key label --rollout-shuffle \
  --custom-rm-path v4probe.reward --rollout-data-postprocess-path v4probe.postprocess \
  --num-rollout $STEPS --rollout-batch-size 8 --n-samples-per-prompt 4 --rollout-max-response-len 32 --rollout-temperature 1 --global-batch-size 32 \
  --tensor-model-parallel-size 1 --context-parallel-size 1 --expert-tensor-parallel-size 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 4096 \
  --advantage-estimator grpo --kl-loss-coef 0.0 --kl-coef 0.0 --entropy-coef 0.0 --eps-clip 0.2 \
  --optimizer adam --lr $LR --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98 \
  --dsa-indexer-loss-coeff 0 $OPT_ARGS \
  --vllm-gpu-memory-utilization 0.5 --vllm-enable-expert-parallel --vllm-moe-backend triton \
  --vllm-kv-cache-dtype fp8 --vllm-block-size 256 --vllm-enable-scale-out --vllm-enforce-eager --vllm-max-model-len 2048 \
  $TOPO_ARGS $EXTRA 2>&1 | tee $RUN/train.log
echo ${PIPESTATUS[0]} > $RUN/rc; touch $RUN/DONE
