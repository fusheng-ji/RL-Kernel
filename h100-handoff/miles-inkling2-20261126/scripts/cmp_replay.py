#!/usr/bin/env python3
# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""Bitwise compare of two trainer-only replays (policy_loss_debug dumps). args: dumpA dumpB"""
import glob
import os
import sys

import torch

A, B = sys.argv[1:3]
fa = {os.path.basename(f): f for f in glob.glob(A + "/policy_loss_debug/*.pt")}
fb = {os.path.basename(f): f for f in glob.glob(B + "/policy_loss_debug/*.pt")}
print("files", len(fa), len(fb), "same set", set(fa) == set(fb))
tot = {k: [0, 0] for k in ("train_log_probs", "old_log_probs", "advantages", "rollout_log_probs")}
for n in fa:
    x, y = torch.load(fa[n], weights_only=False, map_location="cpu"), torch.load(
        fb[n], weights_only=False, map_location="cpu"
    )
    for s, t in zip(x["samples"], y["samples"], strict=True):
        for k in tot:
            if s.get(k) is None:
                continue
            a, b = s[k].flatten().float(), t[k].flatten().float()
            tot[k][0] += a.numel()
            tot[k][1] += int((a.view(torch.int32) != b.view(torch.int32)).sum())
print({k: f"{v[1]}/{v[0]} differ" for k, v in tot.items()})
