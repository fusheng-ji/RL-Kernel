#!/usr/bin/env python3
# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""Token-level comparison of trainer / live SGLang decode / fresh SGLang prefill on the same rollout.
args: <name> <trainer dump dir> <trainer step> <scores.pt or -> ; matches samples by (response_length, rollout_log_probs).
Prints token-weighted mean |d| overall and per |trainer logprob| bucket (same buckets as drift100.py)."""
import glob
import json
import re
import sys

import torch

name, D, step, S = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
B = torch.tensor([1e-4, 1e-2, 0.1, 0.5, 1, 2], dtype=torch.float64)
key = lambda v: torch.as_tensor(v, dtype=torch.float32).flatten().numpy().tobytes()
items = {} if S == "-" else {(i["rl"], key(i["rlp"])): i for i in torch.load(S, weights_only=False)}
acc = {}


def add(k, x, y, ref):
    d = (x.double() - y.double()).abs()
    i = torch.bucketize(ref.double().abs(), B, right=True)
    a = acc.setdefault(
        k, [0, 0.0, 0, torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.float64)]
    )
    a[0] += d.numel()
    a[1] += float(d.sum())
    a[2] += int((x != y).sum())
    a[3] += torch.bincount(i, minlength=7).double()
    a[4] += torch.bincount(i, weights=d, minlength=7)


n = m = 0
for f in glob.glob(D + "/policy_loss_debug/rank_*_call_*.pt"):
    if int(re.search(r"call_(\d+)", f).group(1)) // 32 != step:
        continue
    for s in torch.load(f, weights_only=False, map_location="cpu")["samples"]:
        n += 1
        mk = s["local_loss_mask"].bool().flatten()
        t = s["old_log_probs"].flatten().float()[mk]
        dec = s["rollout_log_probs"].flatten().float()[mk]
        add("trainer_vs_live_decode", t, dec, t)
        it = items.get(
            (
                int(s["response_length"]),
                key(s["rollout_log_probs"].flatten()[-int(s["response_length"]) :]),
            )
        )
        if it is None:
            continue
        m += 1
        pre = torch.tensor(it["prefill"], dtype=torch.float32)[mk]
        add("trainer_vs_fresh_prefill", t, pre, t)
        add("live_decode_vs_fresh_prefill", dec, pre, t)
res = {"name": name, "samples": n, "matched": m}
for k, (nt, sd, ne, bn, bs) in acc.items():
    res[k] = {
        "tokens": nt,
        "mean_abs": sd / nt,
        "ne_frac": ne / nt,
        "bucket_share": (bn / bn.sum()).round(decimals=4).tolist(),
        "bucket_mean_abs": (bs / bn.clamp(min=1)).tolist(),
    }
print(json.dumps(res))
