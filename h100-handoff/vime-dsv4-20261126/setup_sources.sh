#!/bin/bash
# Login node, no GPU. Creates under $W/src:
#   vime          PR head aebf8fea, clean (PR's own smoke tests run here)
#   vime-fix-run  fix branch a2d04354 (3 commits on aebf8fea, from vime-fix.bundle) + patch 0002, uncommitted
#   Megatron-LM   ae76c60e6, clean;  Megatron-run  ae76c60e6 + patch 0004, uncommitted
set -euo pipefail
: "${H:?export H=<THIS_DIR>}"; source $H/env.sh
mkdir -p $W/src; cd $W/src
[ -d vime ] || git clone -q https://github.com/inaniloquentee/vime.git vime
git -C vime fetch -q origin pull/2/head && git -C vime checkout -q --detach aebf8fea6e2ccaf772f54368ad7356f7e90ad11b
git -C vime bundle verify $H/vime-fix.bundle
git -C vime fetch -q $H/vime-fix.bundle fix/dsv4-cuda-grpo:refs/heads/fix/dsv4-cuda-grpo
[ -d vime-fix-run ] || git -C vime worktree add -q --detach $W/src/vime-fix-run a2d04354ed8d2886ed7944d8e1eab437673332c9
git -C vime-fix-run diff --quiet && git -C vime-fix-run apply $H/patches/0002-allow-numpy2-optin.patch
[ -d Megatron-LM ] || git clone -q https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
git -C Megatron-LM checkout -q --detach ae76c60e6
[ -d Megatron-run ] || git -C Megatron-LM worktree add -q --detach $W/src/Megatron-run ae76c60e6
git -C Megatron-run diff --quiet && git -C Megatron-run apply $H/patches/0004-megatron-norm-by-dtype.patch
for d in vime vime-fix-run Megatron-LM Megatron-run; do echo "$d $(git -C $d rev-parse HEAD) diff=$(git -C $d diff | sha256sum | cut -c1-16) $(git -C $d diff --stat | tail -1)"; done
