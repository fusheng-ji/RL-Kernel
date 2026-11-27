#!/bin/bash
S=$SCRATCH_ROOT/sched_20261127; V=$S/verlomni; E2=~/datastorage/venvs/<E2>
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$V/hf; unset HF_TOKEN WANDB_API_KEY
echo "== clone"; [ -d $V/verl-omni ] || git clone -q https://github.com/verl-project/verl-omni.git $V/verl-omni
git -C $V/verl-omni checkout -q --detach 708f4d631b5c77f005fdc112e5de75b52237c5ae; git -C $V/verl-omni rev-parse HEAD; git -C $V/verl-omni status --porcelain --untracked-files=no | wc -l
echo "== overlay (uv pip --no-deps --target; E2 has no pip)"
~/.local/bin/uv pip install --python $E2/bin/python --no-deps --target $V/overlay ftfy==6.3.1 iopath==0.1.10 portalocker==4.4.2 timm==1.0.31 \
  "pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f" \
  "imagebind @ git+https://github.com/facebookresearch/ImageBind.git@53680b02d7e37b19b124fa37bae4b6c98c38f5be"; echo overlay_exit=$?
echo "== rewards"
HF=~/miniconda3/bin/hf
$HF download laion/larger_clap_general --revision ada0c23a36c4e8582805bb38fec3905903f18b41 --local-dir $V/rewards/larger_clap_general; echo clap_exit=$?
$HF download seung275/imagebind_huge.pth --revision b17ac77b71715c0dcefe5600b6a4b25f7cbc8012 --local-dir $V/rewards/_ib; echo ib_exit=$?
ls -la $V/rewards/_ib
echo "== FA3 kernel cache"
PYTHONNOUSERSITE=1 $E2/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('kernels-community/flash-attn3', repo_type='kernel', revision='v1', allow_patterns=['build/torch-stable-abi*-cu130-x86_64-linux/*']))"; echo fa3_exit=$?
echo VERLOMNI_PREP_DONE
