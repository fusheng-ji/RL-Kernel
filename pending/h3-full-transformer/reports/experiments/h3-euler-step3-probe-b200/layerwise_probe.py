"""Layer-by-layer comparison of the step-3 forward (seed 33) for the B200 step-3 spike.

Arms (each compares the video-request row of two forwards; the batched layout is row 1 of a batch of 2):
  A: (single, L_s) vs (batched, L_b)    the natural pair of the full run (the 0.62 spike)
  B: (single, L_s) vs (batched, L_s)    shared latent (control)
  E: (single, L_s) vs (single,  L_b)    same layout, only the latent differs
  P: (single, L_s) vs (single,  L_p)    L_p = L_s with only the elements that differ after the bf16 cast set to L_b's
Also reports how many step-3 input elements differ after the bf16 cast.
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--rl-kernel-repo", type=Path, required=True)
ap.add_argument("--latent-dir", type=Path, required=True)
ap.add_argument("--probe-script", type=Path, required=True)
ap.add_argument("--seed", type=int, default=33)
args = ap.parse_args()
sys.path.insert(0, str(args.rl_kernel_repo.resolve()))
spec = importlib.util.spec_from_file_location("probe", args.probe_script)
probe = importlib.util.module_from_spec(spec); sys.modules["probe"] = probe; spec.loader.exec_module(probe)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from diffusers import MiniMaxH3Transformer3DModel
from rl_engine.validation.models import h3_block
from tools.validation.models.h3_full_model import packed_inputs

model = MiniMaxH3Transformer3DModel.from_pretrained(str(args.checkpoint / "transformer"), torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True).to("cuda").eval()
base = packed_inputs("cuda")
gen = torch.Generator().manual_seed(args.seed)
noise = {"video": torch.randn(base["hidden_states"].shape, generator=gen).cuda(),
         "audio": torch.randn(base["audio_hidden_states"].shape, generator=gen).cuda()}
other = {k: torch.randn(v.shape, generator=torch.Generator().manual_seed(77 + i)).cuda() for i, (k, v) in enumerate(noise.items())}
l_s, l_b = probe.load_latent_pair(args.latent_dir)

# ---- input elements that differ after the bf16 cast
flips = {}
l_p = {k: v.clone() for k, v in l_s.items()}
for m in ("video", "audio"):
    mask = l_s[m].bfloat16() != l_b[m].bfloat16()
    l_p[m][mask] = l_b[m][mask]
    idx = mask.nonzero().tolist()
    flips[m] = {"count": int(mask.sum()), "numel": mask.numel(),
                "fp32_differ": int((l_s[m] != l_b[m]).sum()),
                "max_abs_fp32_delta": float((l_s[m] - l_b[m]).abs().max()),
                "elements": [{"index": i, "single_fp32": float(l_s[m][tuple(i)]), "batched_fp32": float(l_b[m][tuple(i)]),
                              "single_bf16": float(l_s[m].bfloat16()[tuple(i)]), "batched_bf16": float(l_b[m].bfloat16()[tuple(i)])} for i in idx[:8]]}

# ---- capture every node of every block
orig = h3_block.run_graph
store = []
def wrapped(mode, ctx, *a, **k):
    out = orig(mode, ctx, *a, **k)
    if mode == "candidate":
        store.append({n: t.detach().clone() for n, t in out.items() if torch.is_tensor(t)})
    return out
h3_block.run_graph = wrapped
nb = len(model.transformer_blocks)

def run(layout, latent):
    store.clear()
    v = probe.run_step3_velocity(model, base, other, latent=latent, layout=layout, candidate=True)
    assert len(store) == nb, (len(store), nb)
    return v, [dict(s) for s in store]

runs = {"single_Ls": run("single", l_s), "batched_Lb": run("batched", l_b), "batched_Ls": run("batched", l_s),
        "single_Lb": run("single", l_b), "single_Lp": run("single", l_p)}
S = runs["single_Ls"][1][0][h3_block.OUTPUT].shape[1]
groups = {"video": base["video_indices"], "audio": base["audio_indices"], "text": base["text_indices"]}

def row(t, batched):
    if t.dim() == 3 and t.shape[1] == S:
        return t[1:2] if batched else t[0:1]
    return t
def md(x, y): return float((x.float() - y.float()).abs().max())

def compare(run_x, run_y, bx, by):
    (vx, bx_blocks), (vy, by_blocks) = run_x, run_y
    blocks = []
    for i in range(nb):
        o1 = row(bx_blocks[i][h3_block.OUTPUT], bx); o2 = row(by_blocks[i][h3_block.OUTPUT], by)
        d = (o1.float() - o2.float()).abs()
        entry = {"block": i, "out_max_abs": float(d.max()), "out_differing": int((d > 0).sum()), "numel": d.numel(),
                 "out_mean_abs": float(d.mean()), "ref_max_abs": float(o1.float().abs().max()),
                 "rel_l2": float(d.norm() / o1.float().norm().clamp_min(1e-30))}
        for g, ix in groups.items():
            entry[g + "_max_abs"] = float(d.index_select(1, ix).max())
        blocks.append(entry)
    return {"velocity_video_max_abs": md(vx["video"], vy["video"]), "velocity_audio_max_abs": md(vx["audio"], vy["audio"]),
            "velocity_video_ref_max_abs": float(vx["video"].abs().max()), "velocity_audio_ref_max_abs": float(vx["audio"].abs().max()),
            "blocks": blocks}, (bx_blocks, by_blocks)

def nodes_of(blocks_pair, bx, by, i):
    out = {}
    for n in blocks_pair[0][i]:
        if n in blocks_pair[1][i]:
            x, y = row(blocks_pair[0][i][n], bx).float(), row(blocks_pair[1][i][n], by).float()
            if x.shape == y.shape:
                out[n] = {"max_abs": float((x - y).abs().max()), "ref_max_abs": float(x.abs().max()),
                          "rel_l2": float((x - y).norm() / x.norm().clamp_min(1e-30))}
    return out

arms = {"A": ("single_Ls", "batched_Lb", False, True), "B": ("single_Ls", "batched_Ls", False, True),
        "E": ("single_Ls", "single_Lb", False, False), "P": ("single_Ls", "single_Lp", False, False)}
report = {"experiment": "h3_step3_layerwise", "seed": args.seed, "commit": probe.COMMIT,
          "device": torch.cuda.get_device_name(0), "compute_capability": "sm%d%d" % torch.cuda.get_device_capability(),
          "torch": torch.__version__, "step3_input_elements_differing_after_bf16_cast": flips, "arms": {}}
for name, (x, y, bx, by) in arms.items():
    res, pair = compare(runs[x], runs[y], bx, by)
    first = {f"> {th:g}": next((b["block"] for b in res["blocks"] if b["out_max_abs"] > th), None) for th in (1e-5, 1e-3, 1e-2, 1e-1)}
    res["first_block_exceeding"] = first
    res["nodes_block0"] = nodes_of(pair, bx, by, 0)
    res["nodes_block1"] = nodes_of(pair, bx, by, 1)
    report["arms"][name] = res
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "layerwise_report.json").write_text(json.dumps(report, indent=1) + "\n")
print({k: {"vel_video": v["velocity_video_max_abs"], "vel_audio": v["velocity_audio_max_abs"], "first": v["first_block_exceeding"]}
       for k, v in report["arms"].items()})
print({m: (f["count"], f["fp32_differ"]) for m, f in flips.items()})
