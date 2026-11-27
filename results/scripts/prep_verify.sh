#!/usr/bin/env bash
# CPU-only verification of the preparation. Output: checks/prep-verify.txt
source "$(dirname "$0")/common.sh"; G=$SCHED/git/repo.git; O=$SCHED/checks/prep-verify.txt
{
echo "## remote heads (fork) vs README section 1"
git -C $G ls-remote origin 'refs/heads/handoff/h100-*' | awk '{print substr($1,1,9), $2}' | sort -k2
echo "## fetched refs"; git -C $G for-each-ref refs/h100 --format='%(objectname:short=9) %(refname:short)'
echo "## T10 bundle"; git -C $G bundle verify $WT/h_img/h100-handoff/h3-image-t10-20261126/h3-image-t10-20261126.bundle 2>&1 | tail -2
echo "W10 commit $(git -C $WT/W10 rev-parse HEAD)  tree $(git -C $WT/W10 rev-parse 'HEAD^{tree}')"; echo "expected tree: $(cut -d' ' -f3 $WT/h_img/h100-handoff/h3-image-t10-20261126/reference/tree.txt)"
for k in W10 WQ WB WP; do echo "$k $(git -C $WT/$k rev-parse HEAD) tree=$(git -C $WT/$k rev-parse 'HEAD^{tree}') dirty_files=$(git -C $WT/$k status --porcelain --untracked-files=no | wc -l)"; done
echo "pipeline code parent: $(git -C $WT/WP rev-parse HEAD^) (expected 008ca510d...)"
echo "## verl-omni"; git -C $SCHED/verlomni/verl-omni rev-parse HEAD; git -C $SCHED/verlomni/verl-omni status --porcelain --untracked-files=no | wc -l
PYTHONNOUSERSITE=1 PYTHONPATH=$SCHED/verlomni/overlay $E2/bin/python -c "import ftfy, iopath, portalocker, timm, pytorchvideo, imagebind; print('overlay imports ok', timm.__version__)" 2>&1 | tail -2
sha256sum $SCHED/verlomni/rewards/_ib/imagebind_huge.pth | cut -c1-20; echo "expected d6f6c22b..."
ls $SCHED/verlomni/hf/hub/kernels--kernels-community--flash-attn3/snapshots/ ; echo "expected 7cb368cf8278b583132eb72cbf312d54586df2e2 (v1)"
find $SCHED/verlomni/hf/hub/kernels--kernels-community--flash-attn3/snapshots/ -maxdepth 3 | sed "s#$SCHED#<WORK>#" | head -5
echo "## envs"; for e in $E1 $V $E2; do echo "$(basename $e): $($e/bin/python -c 'import torch;print("torch",torch.__version__)' 2>&1 | tail -1)"; done
$V/bin/python -c "import diffusers,accelerate;print('V diffusers',diffusers.__version__,'accelerate',accelerate.__version__)" 2>&1|tail -1
$E2/bin/python -c "import importlib.metadata as m; print({p:m.version(p) for p in ['vllm','vllm-omni','verl','diffusers','transformers']})"
echo "## IMG vae sha256"; sha256sum $IMG/vae_diffusers/diffusion_pytorch_model.safetensors | cut -c1-16; echo "expected 5bb28096..."
echo "## WEIGHTS"; ls -L $WEIGHTS | tr '\n' ' '; echo; ls $WEIGHTS/hf_full/transformer/diffusion_pytorch_model.safetensors.index.json
echo "## T9 run-a for T10-a step c"; ls -d $T9_RUN_A
echo "## miles"; cat $SCHED/miles/runs/h0/image_digest.txt; echo "expected amd64 sha256:0a106222fae2ac646cbd8ca69b2d883dcd1da866b29d0b322e1bcbbbd8f789fa"
python3 - <<PY
import json
ref=json.load(open("$WT/h_mi/h100-handoff/miles-inkling2-20261126/reference/b200-summary.json"))["download_sha256"]
mine={}
for l in open("$SCHED/miles/runs/h0/download.sha256"):
    h,p=l.split(None,1); mine[p.strip()]=h
print("miles download_sha256 equal to reference:", ref==mine, "files", len(mine))
PY
du -sh $SCHED/miles/models $SCHED/miles/datasets | sed "s#$SCHED#<WORK>#"
echo "## vime"; grep -E '^(vime|vime-fix-run|Megatron-LM|Megatron-run) ' $SCHED/checks/vime_prep.log
echo "expected: vime/Megatron-LM diff e3b0c44298fc1c14; vime-fix-run ea9498d46771cd6e; Megatron-run 95f19b25da8b1ff7"
grep -cE ': OK$' $SCHED/checks/vime_prep.log; grep -E 'MISMATCH|FAILED' $SCHED/checks/vime_prep.log || echo "no mismatch lines"
du -sh $SCHED/vime/ckpt/* | sed "s#$SCHED#<WORK>#"
echo "## disk"; df -h $SCRATCH_ROOT | tail -1 | awk '{print "scratch free", $4, "use", $5}'
} > $O 2>&1; cat $O
