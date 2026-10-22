#!/usr/bin/env python3
"""Backward: kernel vs native diffusers vs *deterministic* diffusers, bitwise and vs golden.

Parent runs two workers (native / deterministic) in separate processes, each saving the
gradients of the diffusers replay and of the rl-kernel op for fixed inputs, then compares.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SIZES = {
    "timestep_mlp": (1, 4, 64, 2048),
    "adaln_projection": (1, 4, 64, 2048),
    "adaln_row_gather": (1, 257, 4097, 32768),
    "norm_modulate": (1, 257, 4097, 32768),
    "gate_residual": (1, 257, 4097, 32768),
    "final_adaln_out": (1, 257, 4097, 32768),
}
BUILD = {
    "timestep_mlp": "h3-timestep-mlp-fp32",
    "adaln_projection": "h3-adaln-projection-3mod",
    "adaln_row_gather": "h3-adaln-row-gather",
    "norm_modulate": "h3-rmsnorm",
    "gate_residual": "h3-adaln-gate-residual",
    "final_adaln_out": "h3-final-adaln-out",
}


def worker(a):
    import torch

    sys.path.insert(0, str(Path(a.build) / "tools/validation/models"))
    import h3_prior_art as PA

    torch.backends.cuda.matmul.allow_tf32 = False
    if a.mode == "det":
        torch.use_deterministic_algorithms(True)
    out, meta = {}, {"mode": a.mode, "repeat_equal": {}, "err_vs_golden": {}}
    for cand in PA.candidates(a.op, "plain"):
        name = cand.pop("name")
        if "unavailable" in cand or not cand["grads"]:
            continue
        for n in SIZES[a.op]:
            case = PA._Case(cand["fn"], cand["make"](5, n), cand["rows"], cand["grads"],
                            cand.get("out_dims", (0,)), 5)
            o1, g1 = case.run()
            o2, g2 = case.run()
            key = f"{name}/{n}"
            rep = {k: bool(torch.equal(g1[k], g2[k])) for k in g1 if g1[k] is not None}
            meta["repeat_equal"][key] = rep
            out[key] = {"out": [t.cpu() for t in o1], "grads": {k: v.cpu() for k, v in g1.items() if v is not None}}
            # golden: FP64 autograd of the reference formula with the same upstream grads
            ref = cand.get("ref")
            if ref is not None and n <= 4097:
                try:
                    ins = {k: (v.double().requires_grad_(True) if k in cand["grads"] else v)
                           for k, v in case.inputs.items()}
                    r = ref(**ins)
                    r = r if isinstance(r, (tuple, list)) else (r,)
                    ups = [u.double() for u in case.ups]
                    torch.autograd.backward(list(r), ups)
                    err = {}
                    for k in g1:
                        if g1[k] is None or ins[k].grad is None:
                            continue
                        gd = ins[k].grad
                        err[k] = float((g1[k].double() - gd).abs().max() / gd.abs().max().clamp_min(1e-30))
                    meta["err_vs_golden"][key] = err
                except Exception as exc:  # noqa: BLE001
                    meta["err_vs_golden"][key] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            del case
            torch.cuda.empty_cache()
    torch.save(out, a.out + ".pt")
    Path(a.out + ".json").write_text(json.dumps(meta, indent=1))


def parent(a):
    import torch

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    for mode, env in (("nat", {}), ("det", {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})):
        base = d / f"{a.op}__{mode}"
        if not Path(str(base) + ".json").is_file():
            subprocess.run([sys.executable, __file__, "--worker", "--build", a.build, "--op", a.op,
                            "--mode", mode, "--out", str(base)], env=dict(os.environ, **env), check=True)
    nat = torch.load(str(d / f"{a.op}__nat.pt"))
    det = torch.load(str(d / f"{a.op}__det.pt"))
    mn = json.loads((d / f"{a.op}__nat.json").read_text())
    md = json.loads((d / f"{a.op}__det.json").read_text())
    names = sorted({k.split("/")[0] for k in nat})
    dname = next(n for n in names if n.startswith("diffusers"))
    rows = []
    for n in SIZES[a.op]:
        kd, kk = f"{dname}/{n}", f"rl_kernel/{n}"
        for g in nat[kd]["grads"]:
            ours = det[kk]["grads"].get(g)
            row = {"size": n, "grad": g,
                   "diffusers_native_repeatable": mn["repeat_equal"][kd].get(g),
                   "diffusers_det_repeatable": md["repeat_equal"][kd].get(g),
                   "kernel_repeatable": md["repeat_equal"][kk].get(g),
                   "det_diffusers_equals_native": bool(torch.equal(det[kd]["grads"][g], nat[kd]["grads"][g]))}
            if ours is not None:
                dd = det[kd]["grads"][g]
                row["kernel_equals_det_diffusers"] = bool(torch.equal(ours, dd))
                row["kernel_vs_det_differ_frac"] = float((ours != dd).double().mean())
                row["kernel_vs_det_max_abs"] = float((ours.double() - dd.double()).abs().max())
            row["err_kernel"] = (md["err_vs_golden"].get(kk) or {}).get(g)
            row["err_det_diffusers"] = (md["err_vs_golden"].get(kd) or {}).get(g)
            row["err_native_diffusers"] = (mn["err_vs_golden"].get(kd) or {}).get(g)
            rows.append(row)
    (d / f"{a.op}__summary.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r["size"], r["grad"], "nat_rep", r["diffusers_native_repeatable"], "det_rep", r["diffusers_det_repeatable"],
              "kernel_rep", r["kernel_repeatable"], "k==det", r.get("kernel_equals_det_diffusers"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--build", required=True)
    p.add_argument("--op", required=True)
    p.add_argument("--mode")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    (worker if a.worker else parent)(a)
