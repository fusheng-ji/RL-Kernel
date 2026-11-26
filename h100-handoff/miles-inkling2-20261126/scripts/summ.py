# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""usage: summ.py <report dir> <name>  (reads <name>_curve.json, <name>_e2e.json, <name>_perf.json from collect.py; 100-step run)"""

import json
import sys

import numpy as np

D, NM = sys.argv[1] + "/", sys.argv[2]
c = json.load(open(D + NM + "_curve.json"))
e = json.load(open(D + NM + "_e2e.json"))["step_metrics"]
p = json.load(open(D + NM + "_perf.json"))["samples"]
g = lambda k: np.array([r[k] for r in c], float)
for k in ["reward", "kl", "abs_diff", "loss", "grad_norm", "truncated", "step_time"]:
    v = g(k)
    sl, ic = np.polyfit(np.arange(100), v, 1)
    res = v - (sl * np.arange(100) + ic)
    se = np.sqrt(res.var(ddof=2) / ((np.arange(100) - 49.5) ** 2).sum())
    print(
        f"{k:10s} first10 {v[:10].mean():.6g} last10 {v[-10:].mean():.6g} min {v.min():.6g}@{v.argmin()} max {v.max():.6g}@{v.argmax()} slope*100 {sl*100:.4g} +-{se*100:.3g} nan {np.isnan(v).sum()}"
    )
print(
    "kl out of 0.7-1.2e-3:",
    [(i, round(x * 1e3, 3)) for i, x in enumerate(g("kl")) if not 0.7e-3 <= x <= 1.2e-3],
)
print(
    "absdiff out:",
    [(i, round(x, 5)) for i, x in enumerate(g("abs_diff")) if not 0.012 <= x <= 0.018],
)
t = np.array([r["selected_tokens"] for r in e])
m = np.array([r["bit_mismatches"] for r in e])
f = m / t
print(
    "tok first/last10",
    t[:10].mean(),
    t[-10:].mean(),
    "total",
    t.sum(),
    m.sum(),
    m.sum() / t.sum(),
    "frac first/last",
    f[:10].mean(),
    f[-10:].mean(),
    f.min(),
    f.max(),
)
w = np.array([r["step_seconds"] for r in p])
gt = np.array([r["generated_tokens"] for r in p])
for a, b in [(0, 9), (0, 99), (89, 99)]:
    print(
        "perf",
        a,
        b,
        "step",
        w[a:b].mean(),
        w[a:b].std(ddof=1),
        "tok/s",
        gt[a:b].sum() / w[a:b].sum(),
        *(
            f"{k.split('/')[-1]}={np.mean([r[k] for r in p[a:b]]):.4g}"
            for k in [
                "perf/rollout_time",
                "perf/train_time",
                "perf/actor_train_time",
                "perf/log_probs_time",
                "perf/update_weights_time",
                "perf/tokens_per_gpu_per_sec",
                "prefill_ms",
                "decode_ms",
            ]
        ),
        "mem",
        max(r["peak_memory_bytes"] for r in p[a:b]) / 2**30,
    )
print("total s", w.sum())
print("reward by decade", g("reward").reshape(10, 10).mean(1).round(3))
print("trunc", g("truncated").reshape(10, 10).mean(1).round(3))
print("kl e3", (g("kl") * 1e3).reshape(10, 10).mean(1).round(3))
print("abs", g("abs_diff").reshape(10, 10).mean(1).round(5))
