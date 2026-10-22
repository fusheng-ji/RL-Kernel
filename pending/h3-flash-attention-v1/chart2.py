import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
op = json.load(open("h3_flash_vs_553_op.json"))["rows"]
g = lambda impl, s, k: [x for x in op if x["impl"] == impl and x["seq"] == s][0][k]
seqs = [12, 925, 4096, 8349]; C = {"flash_v1": "C0", "h3_full_553": "C3", "sdpa_provider": "C2"}

# 1: fwd and fwd+bwd grouped bars
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for a, k, t in zip(ax, ("fwd_ms", "fwd_bwd_ms"), ("forward", "forward + backward")):
    for i, impl in enumerate(C):
        v = [g(impl, s, k) for s in seqs]
        b = a.bar(np.arange(4) + i * 0.27, v, 0.27, color=C[impl], label=impl)
        a.bar_label(b, labels=[f"{x:.2f}" if x < 10 else f"{x:.0f}" for x in v], fontsize=7)
    a.set_yscale("log"); a.set_xticks(np.arange(4) + 0.27); a.set_xticklabels(seqs)
    a.set_xlabel("tokens"); a.set_ylabel("ms (H100, log)"); a.set_title(f"Attention operator, {t}")
ax[0].legend(); plt.tight_layout(); plt.savefig("op_bars.png", dpi=130); plt.close()

# 2: speedup of flash over #553 and slowdown vs SDPA
fig, a = plt.subplots(figsize=(6.5, 4.2))
a.plot(seqs, [g("h3_full_553", s, "fwd_bwd_ms") / g("flash_v1", s, "fwd_bwd_ms") for s in seqs], "o-", color="C0", label="flash speedup over #553")
a.plot(seqs, [g("flash_v1", s, "fwd_bwd_ms") / g("sdpa_provider", s, "fwd_bwd_ms") for s in seqs], "s--", color="C2", label="flash slowdown vs SDPA")
a.axhline(1, color="gray", lw=0.8); a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("tokens"); a.set_ylabel("ratio (fwd+bwd)"); a.set_title("Flash vs #553 and vs SDPA"); a.legend()
plt.tight_layout(); plt.savefig("op_ratio.png", dpi=130); plt.close()

# 3: relative error vs FP64
fig, a = plt.subplots(figsize=(7, 4.2)); s3 = [12, 925, 4096]; names = ["out", "dq", "dk", "dv"]
for i, impl in enumerate(C):
    v = [np.mean([g(impl, s, f"rel_{n}") for n in names]) for s in s3]
    a.bar(np.arange(3) + i * 0.27, v, 0.27, color=C[impl], label=impl)
a.set_xticks(np.arange(3) + 0.27); a.set_xticklabels(s3); a.set_xlabel("tokens")
a.set_ylabel("mean rel. error vs FP64 (out, dq, dk, dv)"); a.set_title("Accuracy"); a.legend()
plt.tight_layout(); plt.savefig("op_error.png", dpi=130); plt.close()

# 4: block-0 per-node stacked time at 3 sizes, flash vs #553 vs provider
keys = [("small", "925"), ("4253", "4253"), ("8349", "8349")]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, (k, lab) in zip(ax, keys):
    d = {m: json.load(open(f"nodes_{k}_{m}.json")) for m in ("flash", "h3_full")}
    nodes = [n["node"] for n in d["flash"]["nodes"]]
    cols = {"attention": "C3", "o_proj": "C1", "ffn_gate_up": "C4", "ffn_down": "C5"}
    bars = {"flash": [n["candidate_ms"] for n in d["flash"]["nodes"]],
            "#553": [n["candidate_ms"] for n in d["h3_full"]["nodes"]],
            "provider": [n["provider_ms"] for n in d["flash"]["nodes"]]}
    for x, (name, vals) in enumerate(bars.items()):
        bottom = 0; other = 0
        for nm, v in zip(nodes, vals):
            if nm in cols:
                a.bar(x, v, 0.6, bottom=bottom, color=cols[nm], label=nm if x == 0 else None); bottom += v
            else:
                other += v
        a.bar(x, other, 0.6, bottom=bottom, color="lightgray", label="other nodes" if x == 0 else None)
        a.text(x, bottom + other, f"{bottom + other:.1f}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(range(3)); a.set_xticklabels(bars); a.set_title(f"Block-0 forward, {lab} tokens (ms)")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig("block_stacked.png", dpi=130); plt.close()

# 5: per-node ratio to provider at 4253 (flash vs #553)
d = {m: json.load(open(f"nodes_4253_{m}.json"))["nodes"] for m in ("flash", "h3_full")}
nodes = [n["node"] for n in d["flash"]]; fig, a = plt.subplots(figsize=(13, 4.3))
for i, (m, c) in enumerate((("flash", "C0"), ("h3_full", "C3"))):
    a.bar(np.arange(len(nodes)) + i * 0.4, [n["candidate_ms"] / max(n["provider_ms"], 1e-9) for n in d[m]], 0.4, color=c, label=m)
a.axhline(1, color="gray", lw=0.8); a.set_yscale("log"); a.set_xticks(np.arange(len(nodes)) + 0.2)
a.set_xticklabels(nodes, rotation=45, ha="right"); a.set_ylabel("candidate / provider time (log)")
a.set_title("Per-node slowdown vs provider, block 0, 4253 tokens"); a.legend()
plt.tight_layout(); plt.savefig("block_node_ratio.png", dpi=130); plt.close()
