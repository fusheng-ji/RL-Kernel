#!/usr/bin/env python3
"""RowGather fix vs native / deterministic diffusers vs rl-kernel: repeatability, error, bitwise.

Per op/size: N replays of the diffusers replay with ``index_select`` patched to RowGather ("fix"),
distinct bitwise results; error vs FP64 autograd golden (n <= 4097); bitwise/diff vs rl-kernel grads;
backward wall time of fix vs native vs kernel. Same inputs/upstream grads as the other experiments.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from rowgather_fix import patched_index_select  # noqa: E402

SIZES = {"adaln_row_gather": (1, 257, 4097, 32768), "norm_modulate": (1, 257, 4097, 32768),
         "gate_residual": (1, 257, 4097, 32768), "final_adaln_out": (1, 257, 4097, 32768)}


def h(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]


def timed(f, k=10):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(k): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / k * 1e3


def golden(cand, case):
    ins = {k: (v.double().requires_grad_(True) if k in cand["grads"] else v) for k, v in case.inputs.items()}
    r = cand["ref"](**ins); r = r if isinstance(r, (tuple, list)) else (r,)
    torch.autograd.backward(list(r), [u.double() for u in case.ups])
    return {k: ins[k].grad for k in cand["grads"] if ins[k].grad is not None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build"); ap.add_argument("--op")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--out"); a = ap.parse_args()
    sys.path.insert(0, str(Path(a.build) / "tools/validation/models"))
    import h3_prior_art as PA
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "n_repeats": a.n, "rows": [], "time_ms": {}}
    cands = {c["name"].split("[")[0]: c for c in PA.candidates(a.op, "plain") if "unavailable" not in c}
    dif, ker = cands["diffusers"], cands["rl_kernel"]
    mk = lambda c, n: PA._Case(c["fn"], c["make"](5, n), c["rows"], c["grads"], c.get("out_dims", (0,)), 5)
    for n in SIZES[a.op]:
        cd, ck = mk(dif, n), mk(ker, n)
        kg = ck.run()[1]
        with patched_index_select():
            runs = [cd.run()[1] for _ in range(a.n)]
            # forward bitwise unchanged by the patch
            fo = cd.run()[0]
        nat_out = cd.run()[0]
        fwd_same = all(torch.equal(x, y) for x, y in zip(fo, nat_out))
        gold = golden(dif, cd) if n <= 4097 else {}
        for g in dif["grads"]:
            ts = [r[g] for r in runs if r[g] is not None]
            if not ts: continue
            hs = [h(t) for t in ts]
            kk = kg.get(g)
            row = {"size": n, "grad": g, "distinct_fix": len(set(hs)), "forward_unchanged": fwd_same,
                   "fix_equals_kernel": bool(torch.equal(ts[0], kk)) if kk is not None else None}
            if kk is not None:
                row["fix_vs_kernel_differ_frac"] = float((ts[0] != kk).double().mean())
                row["fix_vs_kernel_max_rel"] = float((ts[0].double() - kk.double()).abs().max() / kk.double().abs().max().clamp_min(1e-30))
            if g in gold:
                gd = gold[g]; den = gd.abs().max().clamp_min(1e-30)
                row["err_fix"] = float((ts[0].double() - gd).abs().max() / den)
                row["err_kernel"] = float((kk.double() - gd).abs().max() / den) if kk is not None else None
                nat = cd.run()[1][g]
                row["err_native_one_run"] = float((nat.double() - gd).abs().max() / den)
            res["rows"].append(row)
        with patched_index_select():
            t_fix = timed(lambda: cd.run())
        res["time_ms"][str(n)] = {"fix": t_fix, "native": timed(lambda: cd.run()), "kernel": timed(lambda: ck.run())}
        del cd, ck; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=1))


main()
