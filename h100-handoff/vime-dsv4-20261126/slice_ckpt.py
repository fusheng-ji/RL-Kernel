"""First-N-layers slice of DeepSeek-V4-Flash-Base (MTP dropped, dimensions unchanged)."""
import json, re, shutil, sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
dst.mkdir(parents=True, exist_ok=False)
wm = json.loads((src/"model.safetensors.index.json").read_text())["weight_map"]
def keep(k):
    if k.startswith("mtp."): return False
    m = re.match(r"layers\.(\d+)\.", k)
    return m is None or int(m.group(1)) < n
by = {}
for k, f in wm.items():
    if keep(k): by.setdefault(f, []).append(k)
out, total = {}, 0
for f, ks in sorted(by.items()):
    with safe_open(src/f, "pt") as h: t = {k: h.get_tensor(k) for k in ks}
    total += sum(v.numel()*v.element_size() for v in t.values())
    save_file(t, dst/f, metadata={"format": "pt"}); out.update({k: f for k in ks})
(dst/"model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": out}))
c = json.loads((src/"config.json").read_text())
c.update(num_hidden_layers=n, compress_ratios=c["compress_ratios"][:n], num_nextn_predict_layers=0)
(dst/"config.json").write_text(json.dumps(c, indent=2))
for x in ("tokenizer.json", "tokenizer_config.json"): shutil.copy(src/x, dst/x)
print("tensors", len(out), "GiB", total/2**30)
