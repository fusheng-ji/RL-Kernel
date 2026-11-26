"""FP8 E4M3 (+ .scale) -> BF16 copy of a V4 checkpoint, with quantization_config removed."""
import json, sys, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=False)
wm = json.loads((src/"model.safetensors.index.json").read_text())["weight_map"]
out, total, cur, idx = {}, 0, {}, 0
files = {}
def get(k):
    f = wm[k]; files.setdefault(f, safe_open(src/f, "pt")); return files[f].get_tensor(k)
def flush():
    global cur, idx
    if not cur: return
    name = f"model-{idx:05d}.safetensors"; save_file(cur, dst/name, metadata={"format": "pt"})
    for k in cur: out[k] = name
    idx += 1; cur = {}
size = 0
for k in sorted(wm):
    if k.endswith(".scale"): continue
    t = get(k)
    if t.dtype == torch.float8_e4m3fn:
        s = get(k.removesuffix(".weight") + ".scale").float(); m, n = t.shape
        br, bc = s.shape
        v = torch.nn.functional.pad(t.float(), (0, bc*128-n, 0, br*128-m)).view(br, 128, bc, 128) * s[:, None, :, None]
        t = v.reshape(br*128, bc*128)[:m, :n].to(torch.bfloat16)
    cur[k] = t.contiguous(); size += t.numel()*t.element_size()
    if size > 4*2**30: total += size; flush(); size = 0
total += size; flush()
(dst/"model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": out}))
c = json.loads((src/"config.json").read_text()); c.pop("quantization_config"); c["expert_dtype"] = "bf16"; c["torch_dtype"] = "bfloat16"
(dst/"config.json").write_text(json.dumps(c, indent=2))
for x in ("tokenizer.json", "tokenizer_config.json"): shutil.copy(src/x, dst/x)
print("tensors", len(out), "GiB", total/2**30)
