#!/usr/bin/env python3
# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""Aggregate a --dump-details policy_loss_debug directory into per-step drift numbers.

Compares, on active (loss_mask) response tokens, in FP32.
"ne" counts numerically different values (0.0 == -0.0); "bit_mismatch" also counts signed-zero encodings.
"mean_abs" is token-weighted; "mean_abs_per_sample" averages per sample first, as the Miles metric does.
Compared pairs:
  rollout_log_probs (SGLang, sampled)  vs  old_log_probs (trainer no-grad scoring pass, same weights)
  train_log_probs (trainer grad pass)  vs  old_log_probs
Usage: analyze_dump.py <dump dir> [--calls-per-step N] [--out out.json]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--calls-per-step", type=int, default=32)
ap.add_argument("--out")
a = ap.parse_args()
files = glob.glob(os.path.join(a.dump, "policy_loss_debug", "rank_*_call_*.pt"))
assert files, "no policy_loss_debug files"
steps = defaultdict(list)
for f in files:
    r, c = map(int, re.search(r"rank_(\d+)_call_(\d+)", f).groups())
    steps[c // a.calls_per_step].append((r, c, f))


def pair(x, y, m):
    x, y = x[m].float(), y[m].float()
    d = (x - y).abs().double()
    bits = x.view(torch.int32) != y.view(torch.int32)
    logr = (y - x).double().clamp(-10, 10)  # rollout->train log ratio
    return d, bits, logr, x != y


report = {}
for s in sorted(steps):
    acc = {"rollout_vs_old": [[], [], [], [], []], "train_vs_old": [[], [], [], [], []]}
    ntok = nsamp = 0
    nonfinite = 0
    keys = []
    for r, c, f in sorted(steps[s]):
        blob = torch.load(f, weights_only=False, map_location="cpu")
        for smp in blob["samples"]:
            nsamp += 1
            m = smp["local_loss_mask"].bool().flatten()
            o = smp["old_log_probs"].flatten()
            t = smp["train_log_probs"].flatten()
            if o.numel() != m.numel():
                raise ValueError(f"mask/logprob length differ in {f}")
            for name, x, y in (
                ("rollout_vs_old", smp.get("rollout_log_probs"), o),
                ("train_vs_old", t, o),
            ):
                if x is None or x.numel() != y.numel():
                    continue
                d, b, lr, ne = pair(x.flatten(), y, m)
                acc[name][0].append(d)
                acc[name][1].append(b)
                acc[name][2].append(lr)
                acc[name][3].append(ne)
                if d.numel():
                    acc[name][4].append(float(d.mean()))
            ntok += int(m.sum())
            nonfinite += int((~torch.isfinite(o)).sum())
    out = {
        "ranks_calls": len(steps[s]),
        "samples": nsamp,
        "active_tokens": ntok,
        "nonfinite_old": nonfinite,
    }
    for name, (d, b, lr, ne, sm) in acc.items():
        if not d:
            continue
        d, b, lr, ne = torch.cat(d), torch.cat(b), torch.cat(lr), torch.cat(ne)
        n = d.numel()
        q = (
            torch.quantile(d, torch.tensor([0.5, 0.99, 0.999], dtype=torch.float64))
            if n < 16_000_000
            else torch.tensor([float("nan")] * 3)
        )
        out[name] = {
            "tokens": n,
            "ne": int(ne.sum()),
            "ne_frac": float(ne.float().mean()),
            "bit_mismatch": int(b.sum()),
            "bit_mismatch_frac": float(b.float().mean()),
            "mean_abs": float(d.mean()),
            "mean_abs_per_sample": sum(sm) / len(sm),
            "p50": float(q[0]),
            "p99": float(q[1]),
            "p999": float(q[2]),
            "max_abs": float(d.max()),
            "kl_k3_mean": float((torch.exp(lr) - 1 - lr).mean()),
        }
    report[s] = out
print(json.dumps(report, indent=1))
if a.out:
    json.dump(report, open(a.out, "w"), indent=1)
