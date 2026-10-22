import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
op = json.load(open("h3_flash_vs_553_op.json"))["rows"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for impl, c in (("flash_v1", "C0"), ("h3_full_553", "C3"), ("sdpa_provider", "C2")):
    r = [x for x in op if x["impl"] == impl]
    ax[0].plot([x["seq"] for x in r], [x["fwd_bwd_ms"] for x in r], "o-", color=c, label=impl)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("tokens"); ax[0].set_ylabel("fwd+bwd ms (H100)")
ax[0].set_title("Attention operator (56 heads, d=128)"); ax[0].legend()
labels = ["925", "4253", "8349"]; keys = ["small", "4253", "8349"]
w = 0.27
for i, (a, c) in enumerate((("flash", "C0"), ("h3_full", "C3"))):
    v = [json.load(open(f"nodes_{k}_{a}.json"))["nodes"] for k in keys]
    ax[1].bar([j + i * w for j in range(3)], [[n for n in nodes if n["node"] == "attention"][0]["candidate_ms"] for nodes in v], w, color=c, label=f"attention node, {a}")
p = [json.load(open(f"nodes_{k}_flash.json"))["nodes"] for k in keys]
ax[1].bar([j + 2 * w for j in range(3)], [[n for n in nodes if n["node"] == "attention"][0]["provider_ms"] for nodes in p], w, color="C2", label="attention node, provider")
ax[1].set_yscale("log"); ax[1].set_xticks([j + w for j in range(3)]); ax[1].set_xticklabels(labels)
ax[1].set_xlabel("tokens"); ax[1].set_ylabel("block-0 attention node ms (fwd)"); ax[1].set_title("Block node"); ax[1].legend()
plt.tight_layout(); plt.savefig("flash_vs_553.png", dpi=130)
