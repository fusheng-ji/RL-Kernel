#!/usr/bin/env python
"""Chart data for the RFC #738 PRs, computed from a checkout of the rebased stack head.

    PYTHONPATH=<checkout> <venv-with-diffusers>/bin/python chart_data.py --out data.json

Everything is recomputed from committed code (no hand-entered numbers). GPU sections need CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch


def fingerprint() -> dict:
    from rl_engine.models.minimax_h3_image import fingerprint as fp

    cfg = fp.transformer_config()
    table = fp.expected_tensor_table(cfg)
    groups: dict[str, int] = defaultdict(int)
    dtypes: dict[str, int] = defaultdict(int)
    for name, (dtype, shape) in table.items():
        n = math.prod(shape)
        head = name.split(".")[0]
        groups[head] += n
        dtypes[dtype] += n
    graph = fp.rows_in_graph()
    return {
        "tensors": len(table),
        "parameters": sum(groups.values()),
        "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
        "dtypes": dict(dtypes),
        "graph_rows": {k: len(v) for k, v in graph.items()},
        "pdd_heads": len(fp.pdd_head_digests()),
        "digest": fp.build_fingerprint()["fingerprint_sha256"],
    }


def position_grid() -> dict:
    from rl_engine.models.minimax_h3_image import position_grid as pg

    rows = []
    for h in range(2, 260, 2):
        for w in (2, 4, 32, 62, 64, 96, 128, 130, 192, 250, 256):
            a = pg.packed_positions(7, [], (h, w)).position_ids
            b = pg.packed_positions(7, [], (h, w), axis="torch").position_ids
            d64 = float((a - b).abs().max())
            n64 = int((a != b).any(dim=1).sum())
            n32 = int((a.float() != b.float()).any(dim=1).sum())
            rows.append({"h": h, "w": w, "max_abs_fp64": d64, "rows_fp64": n64, "rows_fp32": n32})
    return {
        "canvases": len(rows),
        "drift_fp64": sum(r["rows_fp64"] > 0 for r in rows),
        "drift_fp32": sum(r["rows_fp32"] > 0 for r in rows),
        "max_abs_fp64": max(r["max_abs_fp64"] for r in rows),
        "rows": rows,
    }


def canvas() -> dict:
    from rl_engine.models.minimax_h3_image import canvas as cv

    sizes = [256, 384, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096]
    grid = []
    for h in sizes:
        for w in sizes:
            plan = cv.resolve_canvas(h, w)
            grid.append(
                {
                    "h": h,
                    "w": w,
                    "canvas": list(plan.canvas),
                    "scale": plan.scale,
                    "resize": plan.needs_resize,
                    "area_ratio": plan.canvas[0] * plan.canvas[1] / (h * w),
                }
            )
    return {"min_generation_area": cv.DEFAULT_MIN_GENERATION_AREA, "grid": grid}


def _time(fn, repeats=20):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1e3)
    return float(np.median(out))


def patch_pack() -> dict:
    from rl_engine.models.minimax_h3_image import patch_pack as pp

    packer = pp.ImagePatchPacker()
    out = []
    for h, w in ((64, 64), (128, 128), (222, 296), (256, 256)):
        z = torch.randn(1, 24, h, w, device="cuda")
        rows = packer.pack_image(z)
        out.append(
            {
                "latent": [h, w],
                "rows": int(rows.shape[1]),
                "pack_ms": _time(lambda: packer.pack_image(z)),
                "unpack_ms": _time(lambda: packer.unpack_image(rows, (h, w))),
                "round_trip_bitwise": bool(torch.equal(packer.unpack_image(rows, (h, w)), z)),
            }
        )
    return {"cases": out}


def velocity() -> dict:
    from rl_engine.backends.shared.triton.flow.velocity_step import H3ImageVelocityStepTritonOp
    from rl_engine.models.minimax_h3_image import pdd_schedule as ps
    from rl_engine.reference.flow.ode_step import NativeH3OdeStepOp
    from rl_engine.reference.flow.velocity_step import NativeH3ImageVelocityStepOp

    sched = ps.released_schedule()
    ref, tri, blend = NativeH3ImageVelocityStepOp(), H3ImageVelocityStepTritonOp(), NativeH3OdeStepOp()
    g = torch.Generator().manual_seed(0)
    rows = 16384  # a 4096x4096 canvas: 128 x 128 patches
    x = torch.randn(rows, 96, generator=g).cuda()
    steps = []
    for i in range(6):
        v = torch.randn(rows, 96, generator=g).cuda()
        s, sn = sched.sigmas[i].cuda(), sched.sigmas[i + 1].cuda()
        a = ref(x, v, s, sn)
        b = tri(x, v, s, sn)
        c, _ = blend(x, v, s, sn)
        ulp = (a.view(torch.int32).long() - c.view(torch.int32).long()).abs()
        steps.append(
            {
                "step": i,
                "sigma": float(s),
                "sigma_next": float(sn),
                "triton_vs_reference_differing": int((a.view(torch.int32) != b.view(torch.int32)).sum()),
                "blend_vs_velocity_differing": int((ulp > 0).sum()),
                "blend_vs_velocity_max_abs": float((a - c).abs().max()),
                "max_abs_value": float(a.abs().max()),
                "elements": a.numel(),
                "reference_ms": _time(lambda: ref(x, v, s, sn)),
                "triton_ms": _time(lambda: tri(x, v, s, sn)),
            }
        )
        x = a
    return {"rows": rows, "steps": steps}


def pdd_schedule() -> dict:
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    from rl_engine.models.minimax_h3_image import pdd_schedule as ps
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    ours = ps.released_schedule()
    h3 = NativeH3SigmaScheduleOp()(7, shift=12.0)
    shifted = MiniMaxH3Scheduler(shift_provided_sigmas=True, euler_update_form="velocity")
    shifted.set_timesteps(sigmas=ours.sigmas)
    return {
        "released": [float(s) for s in ours.sigmas],
        "h3_shift12_linspace7": [float(s) for s in h3.sigmas],
        "shift_provided_sigmas": [float(s) for s in shifted.sigmas],
        "timesteps": [float(t) for t in ours.timesteps],
        "round_trip_moved": ps.round_trip_moved(ps.config_sigmas()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = {
        "fingerprint": fingerprint(),
        "position_grid": position_grid(),
        "canvas": canvas(),
        "pdd_schedule": pdd_schedule(),
        "environment": {"torch": torch.__version__},
    }
    if torch.cuda.is_available():
        data["environment"]["gpu"] = torch.cuda.get_device_name(0)
        data["patch_pack"] = patch_pack()
        data["velocity"] = velocity()
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
