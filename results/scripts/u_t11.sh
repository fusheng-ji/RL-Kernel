#!/usr/bin/env bash
# T11: T10-edit run-b (--strict-only --against run-a on results branch).
D=$(dirname "$0"); source "$D/common.sh"
J=${SLURM_JOB_ID:?}
NB=$(nodes_of "$J" | sed -n 2p)
O=$OUT/T11
mkdir -p "$O"
git -C "$W10" status --porcelain | wc -l >"$O/status-before.txt"
[ "$(cat "$O/status-before.txt" | tr -d ' ')" = 0 ] || { log "STOP: W10 dirty"; exit 1; }
idle_gpu=; for g in 0 1 2 3; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  [ "$used" = "0" ] && idle_gpu=$g && break
done
[ -n "$idle_gpu" ] || { log "STOP: T11 no idle GPU on node-b"; exit 1; }
echo "GPU=$idle_gpu node=$NB" >"$O/gpu-card.txt"
mkdir -p "$O/against"
git -C "$SCHED/wt/push" fetch origin handoff/h100-h3-image-results-20261123
ref=$(git -C "$SCHED/wt/push" rev-parse FETCH_HEAD)
echo "against_ref=$ref" >"$O/against-ref.txt"
git -C "$SCHED/wt/push" archive "FETCH_HEAD" results/T10/edit/run-a | tar -x -C "$O/against"
export RL_KERNEL_H3_IMAGE_ROOT=$IMG RL_KERNEL_H3_IMAGE_DIR=$IMG RL_KERNEL_H3_IMAGE_VAE_DIR=$IMG/vae_diffusers
export RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=$CKPT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
cd "$W10"
nvidia-smi >"$O/nvidia-smi.txt"
$V/bin/pip list 2>/dev/null | grep -E '^(torch|triton|diffusers|transformers|accelerate) ' >"$O/env.txt"
env CUDA_VISIBLE_DEVICES="$idle_gpu" PYTHONPATH="$W10" $V/bin/python tools/validation/models/h3_image_trajectory.py \
  --root "$IMG" --text-encoder-root "$CKPT" --out "$O/run-b" \
  --fixtures edit_two_refs,edit_example --strict-only \
  --against "$O/against/results/T10/edit/run-a" >"$O/run-b.log" 2>&1
echo "exit=$?" >>"$O/run-b.log"
git status --porcelain | wc -l >"$O/status-after.txt"
grep -q 'exit=0' "$O/run-b.log" && grep -q 'ok True' "$O/run-b.log" || { log "STOP: T11 run-b failed"; exit 1; }
log "T11 run-b done GPU=$idle_gpu"