import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
r = json.load(open("bitwise_check.json"))
impls = ["flash_v1", "h3_full_553", "sdpa_provider"]; sc = ["repeat_x20", "batch3_vs_1", "head_shard_14of56"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
a = ax[0]; M = np.array([[r[i][s]["mismatch_frac"] for s in sc] for i in impls])
a.imshow(np.where(M == 0, 0, 1), cmap=matplotlib.colors.ListedColormap(["#b7e4c7", "#ffd6a5"]), vmin=0, vmax=1, aspect="auto")
for i in range(3):
    for j in range(3):
        a.text(j, i, "bitwise equal\n(0 mismatches)" if M[i, j] == 0 else f"{M[i, j]:.1e} of elements\ndiffer (worst tensor)", ha="center", va="center", fontsize=9)
a.set_xticks(range(3)); a.set_xticklabels(["same input,\n20 repeats", "batch 3 vs batch 1", "14 of 56 heads\n(shard) vs full"])
a.set_yticks(range(3)); a.set_yticklabels(impls); a.set_title(f"Self-consistency, {r['seq']} tokens, out/dq/dk/dv")
a = ax[1]; keys = ["flash_vs_553", "flash_vs_sdpa", "553_vs_sdpa"]
v = [r["cross_backend"][k]["mismatch_frac"] for k in keys]
b = a.bar(keys, v, color=["C1", "C4", "C5"]); a.bar_label(b, labels=[f"{x:.1%}" for x in v])
a.set_ylabel("fraction of elements that differ (worst tensor)"); a.set_title("Across backends (expected to differ)")
plt.tight_layout(); plt.savefig("bitwise_consistency.png", dpi=130)
