"""Bitwise consistency of flash vs #553 vs SDPA: repeat, batch, head shard, cross-backend (H100)."""
import json, sys
import torch
sys.path.insert(0, ".")
from rl_engine.backends.shared.triton.attention.h3_flash import H3FlashAttentionOp
from rl_engine.backends.cuda.attention.h3_full_attention import H3FullAttentionOp

H, D, S = 56, 128, 925
def sdpa(q, k, v):
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2)
OPS = {"flash_v1": H3FlashAttentionOp(), "h3_full_553": H3FullAttentionOp(), "sdpa_provider": sdpa}

def run(op, q, k, v, g):
    q, k, v = (x.detach().clone().requires_grad_() for x in (q, k, v))
    o = op(q, k, v); o.backward(g)
    return o.detach(), q.grad, k.grad, v.grad

def diff(a, b):
    n = a.numel(); m = (a != b)
    return {"mismatch_frac": m.sum().item() / n,
            "max_abs": (a.float() - b.float()).abs().max().item()}

def worst(xs, ys):
    ds = [diff(x, y) for x, y in zip(xs, ys)]
    return {"mismatch_frac": max(d["mismatch_frac"] for d in ds), "max_abs": max(d["max_abs"] for d in ds),
            "per_tensor": dict(zip(("out", "dq", "dk", "dv"), (d["mismatch_frac"] for d in ds)))}

gen = torch.Generator(device="cuda").manual_seed(7)
mk = lambda b: torch.randn(b, S, H, D, device="cuda", dtype=torch.bfloat16, generator=gen)
q3, k3, v3, g3 = mk(3), mk(3), mk(3), mk(3)
q1, k1, v1, g1 = (x[:1].contiguous() for x in (q3, k3, v3, g3))
res = {}
for name, op in OPS.items():
    r = {}
    base = run(op, q1, k1, v1, g1)
    # repeat: worst over 20 repeats
    reps = [worst(base, run(op, q1, k1, v1, g1)) for _ in range(20)]
    r["repeat_x20"] = max(reps, key=lambda d: d["mismatch_frac"])
    # batch invariance: sample i of a batch-3 run vs batch-1 run of the same sample
    bat = run(op, q3, k3, v3, g3); bs = []
    for i in range(3):
        single = run(op, *(x[i:i + 1].contiguous() for x in (q3, k3, v3, g3)))
        bs.append(worst([t[i:i + 1] for t in bat], single))
    r["batch3_vs_1"] = max(bs, key=lambda d: d["mismatch_frac"])
    # head shard: 14 heads zero-padded vs the same heads in the full 56-head run
    pad = []
    for x in (q1, k1, v1, g1):
        p = torch.zeros_like(x); p[:, :, :14] = x[:, :, :14]; pad.append(p)
    sh = run(op, *pad)
    r["head_shard_14of56"] = worst([t[:, :, :14] for t in base], [t[:, :, :14] for t in sh])
    res[name] = r
res["cross_backend"] = {
    "flash_vs_553": worst(run(OPS["flash_v1"], q1, k1, v1, g1), run(OPS["h3_full_553"], q1, k1, v1, g1)),
    "flash_vs_sdpa": worst(run(OPS["flash_v1"], q1, k1, v1, g1), run(OPS["sdpa_provider"], q1, k1, v1, g1)),
    "553_vs_sdpa": worst(run(OPS["h3_full_553"], q1, k1, v1, g1), run(OPS["sdpa_provider"], q1, k1, v1, g1)),
}
res["device"] = torch.cuda.get_device_name(); res["seq"] = S
json.dump(res, open("bitwise_check.json", "w"), indent=1)
for k, v in res.items():
    if isinstance(v, dict): print(k, json.dumps({a: (b["mismatch_frac"] if "mismatch_frac" in b else {c: d["mismatch_frac"] for c, d in b.items()}) for a, b in v.items()}))
