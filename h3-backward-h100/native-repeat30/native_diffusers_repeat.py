#!/usr/bin/env python3
"""Native-diffusers backward repeatability, many repeats, with attribution.

For each op/size: N backward replays of diffusers (native), diffusers (deterministic mode) and the
rl-kernel op on identical inputs; counts distinct bitwise results per gradient, spread between
runs (relative to max|grad|), and lists the CUDA kernels of one native backward.
"""
import argparse, hashlib, json, sys, warnings
from pathlib import Path
import torch

SIZES = {"timestep_mlp": (1, 4, 64, 2048), "adaln_projection": (1, 4, 64, 2048),
         "adaln_row_gather": (1, 257, 4097, 32768), "norm_modulate": (1, 257, 4097, 32768),
         "gate_residual": (1, 257, 4097, 32768), "final_adaln_out": (1, 257, 4097, 32768)}

def h(t): return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build"); ap.add_argument("--op")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--out"); ap.add_argument("--mode", choices=["native","det"])
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.build) / "tools/validation/models"))
    import h3_prior_art as PA
    from torch.profiler import profile, ProfilerActivity
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "n_repeats": a.n, "rows": []}
    for cand in PA.candidates(a.op, "plain"):
        name = cand.pop("name")
        if "unavailable" in cand or not cand["grads"]: continue
        kind = "diffusers" if name.startswith("diffusers") else name
        if kind not in ("diffusers", "rl_kernel"): continue
        if kind == "rl_kernel" and a.mode == "det": continue
        for mode in ((a.mode,) if kind == "diffusers" else ("native",)):
            for n in SIZES[a.op]:
                torch.use_deterministic_algorithms(mode == "det", warn_only=True)
                case = PA._Case(cand["fn"], cand["make"](5, n), cand["rows"], cand["grads"], cand.get("out_dims", (0,)), 5)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    runs = [case.run()[1] for _ in range(a.n)]
                warn = sorted({str(x.message).split(".")[0][:140] for x in w})
                kern = []
                if mode == "native":
                    with profile(activities=[ProfilerActivity.CUDA]) as p: case.run(); torch.cuda.synchronize()
                    kern = sorted({e.name[:90] for e in p.events() if e.device_type.name == "CUDA"})
                for g in cand["grads"]:
                    ts = [r[g] for r in runs if r[g] is not None]
                    if not ts: continue
                    hs = [h(t) for t in ts]
                    ref = ts[0].double(); sc = ref.abs().max().clamp_min(1e-30)
                    spread = max(float((t.double() - ref).abs().max() / sc) for t in ts)
                    res["rows"].append({"impl": kind, "mode": mode, "size": n, "grad": g, "distinct": len(set(hs)),
                        "max_rel_spread": spread, "first_hash": hs[0], "warn": warn, "kernels": kern})
                del case; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=1))
main()
