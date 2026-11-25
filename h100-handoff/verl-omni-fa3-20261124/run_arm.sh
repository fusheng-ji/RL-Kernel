#!/usr/bin/env bash
# Stock verl-omni H3 T2VA FlowGRPO (708f4d63) through its own V1 launcher, one node, 4 GPUs, rollout TP4, actor DP4.
#   ARM=sdpa|fa3  WL=w0|w1  CAP=0|1  STEPS=<n>  TAG=<name>  bash run_arm.sh
# Paths: E2 (venv), VERL_OMNI (checkout), CKPT (holds FL2VA/ and transformer/), REWARDS (holds larger_clap_general/ and
# imagebind_huge.pth), OVERLAY (the reward packages), OUT (results).
set -uo pipefail
: "${ARM:?}" "${WL:?}" "${CAP:?}" "${STEPS:?}" "${TAG:?}" "${E2:?}" "${VERL_OMNI:?}" "${CKPT:?}" "${REWARDS:?}" "${OUT:?}"
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case $ARM in
  sdpa) ACTOR=native; ROLLOUT=TORCH_SDPA ;;
  fa3)  ACTOR=_flash_3_varlen_hub; ROLLOUT=FLASH_ATTN_3_HUB ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
WLARGS=()
case $WL in
  w0) ;;  # the recipe's training shape: 256x384, 121 frames, CPS window 3
  w1) export HEIGHT=768 WIDTH=1344 NUM_FRAMES=107
      WLARGS=(actor_rollout_ref.rollout.algo.noise_level=0.7 actor_rollout_ref.rollout.algo.sde_type=sde
              "actor_rollout_ref.rollout.algo.sde_window_range=[1,4]" actor_rollout_ref.rollout.algo.sde_window_size=2
              actor_rollout_ref.actor.optim.lr=1e-4) ;;
  *) echo "unknown WL=$WL"; exit 2 ;;
esac
RUN=$OUT/runs/$WL-$ARM-cap$CAP-$TAG
[[ -e $RUN ]] && { echo "$RUN exists"; exit 2; }
mkdir -p "$RUN"
dirty() { git -C "$VERL_OMNI" status --porcelain --untracked-files=no | wc -l; }
git -C "$VERL_OMNI" rev-parse HEAD > "$RUN/commit.txt"
[[ $(dirty) == 0 ]] || { echo "verl-omni tree is dirty" | tee "$RUN/summary.txt"; exit 3; }
export TMPDIR=${SHORT_TMP:-/tmp}/vf$$; mkdir -p "$TMPDIR"   # short: vLLM's IPC socket paths
trap 'rm -rf "${TMPDIR:?}"' EXIT
export RAY_TMPDIR=$TMPDIR PATH=$E2/bin:$PATH PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 USE_FLAX=0
export WANDB_MODE=disabled OMP_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH=$VERL_OMNI:$H/h3cap${OVERLAY:+:$OVERLAY}
[[ $CAP == 1 ]] && export H3CAP_DIR=$RUN/cap
export MODEL_PATH=$CKPT/FL2VA ACTOR_CONFIG_PATH=$CKPT/transformer DATA_DIR=$H/prompts
export CLAP_MODEL_PATH=$REWARDS/larger_clap_general IMAGEBIND_MODEL_PATH=$REWARDS/imagebind_huge.pth OUTPUT_DIR=$RUN/out
export NUM_GPUS=4 ROLLOUT_TP=4 ACTOR_SP=1 TOTAL_TRAINING_STEPS=$STEPS INFER_STEPS=10
nvidia-smi > "$RUN/nvidia-smi-start.txt"
python - > "$RUN/env.txt" 2>&1 <<'PY'
import os, torch, diffusers, vllm, vllm_omni, verl_omni, transformers, kernels
print("torch", torch.__version__, "diffusers", diffusers.__version__, "vllm", vllm.__version__, "vllm_omni", vllm_omni.__version__,
      "transformers", transformers.__version__, "kernels", kernels.__version__)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), "gpus", torch.cuda.device_count())
print("verl_omni under VERL_OMNI:", os.path.realpath(verl_omni.__file__).startswith(os.path.realpath(os.environ["PYTHONPATH"].split(":")[0]) + "/"))
PY
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 > "$RUN/smi.csv" & smi=$!
cd "$VERL_OMNI"
timeout "${TLIM:-14000}" bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_v1.sh \
  data.train_batch_size=2 actor_rollout_ref.rollout.n=4 actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.model.attn_backend=$ACTOR actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT \
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.trust_remote_code=True \
  'trainer.logger=["console"]' trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 \
  trainer.log_val_generations=0 trainer.rollout_data_save_freq=1 "${WLARGS[@]}" > "$RUN/train.log" 2>&1
echo "train exit=$?" >> "$RUN/summary.txt"
kill $smi 2>/dev/null
awk -F', ' '{if ($2>m[$1]) m[$1]=$2} END {for (i in m) printf "gpu%s_peak_MiB %s\n", i, m[i]}' "$RUN/smi.csv" | sort >> "$RUN/summary.txt"
grep -oE 'step:[0-9]+ - .*' "$RUN/train.log" > "$RUN/steps.txt"
echo "verl-omni dirty files after: $(dirty)" >> "$RUN/summary.txt"
cat "$RUN/summary.txt"
