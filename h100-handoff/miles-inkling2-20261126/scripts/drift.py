# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""usage: drift.py <dump dir> <rows out .pt>
Why does the rollout-vs-trainer abs diff move over 100 steps? Per-step stats split by response length and by token confidence."""

import glob
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

D, OUT = sys.argv[1], sys.argv[2]
B = [0, 1e-4, 1e-2, 0.1, 0.5, 1, 2, 1e9]  # bins on |trainer old logprob|


def one(args):
    s, f = args
    b = torch.load(f, weights_only=False, map_location="cpu")
    out = []
    for x in b["samples"]:
        m = x["local_loss_mask"].bool().flatten()
        r = x["rollout_log_probs"].flatten()[m].double()
        o = x["old_log_probs"].flatten()[m].double()
        if not o.numel():
            continue
        d = (r - o).abs()
        lr = (o - r).clamp(-10, 10)
        k = torch.exp(lr) - 1 - lr
        i = torch.bucketize(o.abs(), torch.tensor(B[1:-1], dtype=torch.float64), right=True)
        n = torch.bincount(i, minlength=len(B) - 1)
        sd = torch.bincount(i, weights=d, minlength=len(B) - 1)
        sk = torch.bincount(i, weights=k, minlength=len(B) - 1)
        out.append(
            (
                s,
                o.numel(),
                float(d.mean()),
                float(k.mean()),
                float(-o.mean()),
                n.tolist(),
                sd.tolist(),
                sk.tolist(),
            )
        )
    return out


if __name__ == "__main__":
    torch.set_num_threads(1)
    jobs = []
    for f in glob.glob(D + "/policy_loss_debug/rank_*_call_*.pt"):
        c = int(re.search(r"call_(\d+)", f).group(1))
        jobs.append((c // 32, f))
    with ProcessPoolExecutor(min(24, os.cpu_count() or 4)) as ex:
        rows = [r for l in ex.map(one, jobs, chunksize=64) for r in l]
    rows.sort(key=lambda r: r[0])
    torch.save(rows, OUT)
    s = np.array([r[0] for r in rows])
    n = np.array([r[1] for r in rows])
    d = np.array([r[2] for r in rows])
    k = np.array([r[3] for r in rows])
    e = np.array([r[4] for r in rows])
    N = np.array([r[5] for r in rows])
    SD = np.array([r[6] for r in rows])
    for a in range(0, 100, 10):
        q = (s >= a) & (s < a + 10)
        z = N[q].sum(0)
        bins = SD[q].sum(0) / np.maximum(z, 1)
        L = [
            f"{d[q&(n>=l)&(n<h)].mean():.4f}({(q&(n>=l)&(n<h)).sum()})"
            for l, h in [(0, 500), (500, 1500), (1500, 3000), (3000, 9999)]
        ]
        print(
            f"{a:3d} samples {q.sum()} tok {n[q].mean():.0f} per-sample {d[q].mean():.5f} token-weighted {(d[q]*n[q]).sum()/n[q].sum():.5f} nll {(e[q]*n[q]).sum()/n[q].sum():.4f} len-bins {L} conf-share {np.round(z/z.sum(),3)} conf-mean {np.round(bins,5)}"
        )
    # decomposition last20 vs first20, token weighted
    q1, q2 = s < 20, s >= 80
    p1, p2 = N[q1].sum(0) / N[q1].sum(), N[q2].sum(0) / N[q2].sum()
    m1, m2 = SD[q1].sum(0) / N[q1].sum(0), SD[q2].sum(0) / N[q2].sum(0)
    print(
        "token-weighted total",
        (p1 * m1).sum(),
        (p2 * m2).sum(),
        "mix effect",
        ((p2 - p1) * m1).sum(),
        "within-bin",
        (p2 * (m2 - m1)).sum(),
        "within at p1",
        (p1 * (m2 - m1)).sum(),
    )
