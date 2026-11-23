#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Cross-environment digests of the RFC #738 H3-Image CPU/CUDA rows (H100 handoff, 2026-11-23).

Run from a checkout of the integrated head; writes one JSON with no paths:

    PYTHONPATH=<W> python h3_image_cross_env_probe.py --out probe.json [--compare probe-b200.json]

Recorded, each as a digest of the exact bytes produced:

* ``fingerprint``: the architecture fingerprint digest (CPU, stdlib only).
* ``position_grid``: float64 and float32 grids of 1,419 T2I canvases and 6 edit layouts.
* ``canvas``: 46,117 (height, width, area) plans, and the PIL LANCZOS resize bytes of five
  requests. A plan that resizes records the Pillow version in its artifact, so ``plans``
  (the artifact digests) is Pillow-dependent; ``plans_geometry`` hashes the same artifacts
  without the ``pillow`` field and must match everywhere (H100 round 1, 2026-11-23: Pillow
  13.0.0 vs 12.3.0 moved ``plans`` and nothing else).
* ``packed_layout``: three layouts' artifacts and a permutation's row-identity digest.
* ``patch_pack``: rows of 5 shapes x 3 dtypes on CPU and on CUDA, and their round trip.
* ``trajectory``: the sealed artifact of the test loop (T2I and edit) on CPU.
* ``permutation_drift`` (needs Diffusers 875373ac): tiny float64 / float32 transformer on
  CUDA, max |output difference| over 20 consistent permutations. Hardware-dependent value;
  reported, not compared.

``--compare`` prints which digests differ from a reference JSON and exits 1 if any
hardware-independent digest differs. Differences in Pillow-dependent bytes are reported
separately, with both Pillow versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_engine.models.minimax_h3.trajectory import tensor_digest
from rl_engine.models.minimax_h3_image import canvas as cv
from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import patch_pack as pp
from rl_engine.models.minimax_h3_image import position_grid as pg

SCHEMA = "rlkernel.h3_image.cross_env_probe.v1"
PILLOW_DEPENDENT = ("canvas.resize", "canvas.plans")
REPORT_ONLY = ("permutation_drift", "environment")


def _sha(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def environment() -> dict[str, Any]:
    import PIL

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env.update(
            device_name=torch.cuda.get_device_name(0),
            compute_capability=list(torch.cuda.get_device_capability(0)),
            cuda=torch.version.cuda,
        )
    try:
        import diffusers

        env["diffusers"] = diffusers.__version__
    except ImportError:
        env["diffusers"] = None
    return env


def position_grid() -> dict[str, str]:
    t2i = []
    for h in range(2, 260, 2):
        for w in (2, 4, 32, 62, 64, 96, 128, 130, 192, 250, 256):
            art = pg.packed_positions(7, [], (h, w)).artifact()
            t2i.append(f"{h}x{w}|{art['sha256_fp64']}|{art['sha256_fp32']}")
    refs = [(64, 64), (48, 80), (126, 2), (2, 126), (250, 98)]
    edit = []
    for k in range(len(refs) + 1):
        art = pg.packed_positions(333, refs[:k], (96, 64)).artifact()
        edit.append(f"{k}|{art['sha256_fp64']}|{art['sha256_fp32']}")
    return {"t2i_1419": _sha(t2i), "edit_6": _sha(edit), "cases": str(len(t2i) + len(edit))}


def canvas() -> dict[str, Any]:
    plans = []
    geometry = []

    def record(plan: cv.CanvasPlan) -> None:
        art = plan.artifact()
        plans.append(art["sha256"])
        body = {k: v for k, v in art.items() if k not in ("pillow", "sha256")}
        geometry.append(json.dumps(body, sort_keys=True))
    areas = [1024, 512 * 512, 1024 * 1024, 1536 * 1536, 2048 * 2048, 4096 * 4096, 1000003]
    sides = list(range(1, 300)) + list(range(300, 4200, 7)) + [4095, 4096, 4097, 8191]
    for a in areas:
        for h in sides[::3]:
            for w in (1, 17, 32, 48, 80, 100, 512, 777, 1024, 2047, 2048, 3000, 4096):
                record(cv.resolve_canvas(h, w, a))
    rng = random.Random(0)
    for _ in range(20000):
        h, w = rng.randint(1, 6000), rng.randint(1, 6000)
        record(cv.resolve_canvas(h, w, rng.randint(1024, 4096 * 4096)))
    resize = {}
    generator = torch.Generator().manual_seed(0)
    for h, w, a in ((80, 80, 1024), (1000, 700, 4194304), (48, 1000, 1024), (333, 517, 1048576)):
        plan = cv.resolve_canvas(h, w, a)
        images = (torch.rand(2, 3, *plan.canvas, generator=generator) * 255).to(torch.uint8)
        resize[f"{h}x{w}@{a}"] = tensor_digest(cv.resize_output(images, plan))
    return {
        "plans": _sha(plans),
        "plans_geometry": _sha(geometry),
        "cases": len(plans),
        "resize": resize,
    }


def packed_layout() -> dict[str, str]:
    out = {}
    for name, (num_text, refs, target) in {
        "t2i": (5, [], (4, 4)),
        "edit2": (9, [(4, 4), (2, 6)], (4, 6)),
        "edit3": (40, [(30, 30), (6, 62), (2, 2)], (62, 98)),
    }.items():
        tags = torch.ones(num_text, dtype=torch.long)
        tags[1 : min(4, num_text)] = 0 if refs else 1
        layout = pl.build_layout(tags, refs, target)
        art = layout.artifact()
        order = torch.randperm(layout.sequence_length, generator=torch.Generator().manual_seed(7))
        permuted = pl.permute(layout, order)
        out[name] = _sha(
            [art["layout_sha256"], art["row_identity_sha256"], pl.layout_digest(permuted)]
        )
        out[f"{name}.row_identity_preserved"] = str(
            pl.row_identity_digest(permuted) == art["row_identity_sha256"]
        )
    return out


def patch_pack() -> dict[str, str]:
    packer = pp.ImagePatchPacker()
    out: dict[str, str] = {}
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for h, w in ((2, 2), (4, 6), (62, 98), (128, 128), (6, 250)):
            generator = torch.Generator().manual_seed(h * 1000 + w)
            latents = torch.randn(2, 24, h, w, generator=generator).to(dtype)
            key = f"{str(dtype).split('.')[-1]}.{h}x{w}"
            cpu = packer.pack_image(latents)
            out[f"{key}.cpu"] = tensor_digest(cpu)
            if torch.cuda.is_available():
                gpu = packer.pack_image(latents.cuda())
                back = packer.unpack_image(gpu, (h, w))
                out[f"{key}.cuda_equals_cpu"] = str(tensor_digest(gpu) == out[f"{key}.cpu"])
                out[f"{key}.cuda_round_trip"] = str(tensor_digest(back) == tensor_digest(latents))
    return out


def trajectory(test_dir: Path) -> dict[str, str]:
    sys.path.insert(0, str(test_dir))
    import test_h3_image_trajectory as t  # noqa: E402

    return {task: t.run(task)[0]["artifact_sha256"] for task in ("t2i", "edit")}


def permutation_drift(test_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"skipped": "no CUDA"}
    try:
        from diffusers.models.transformers import MiniMaxH3ImageTransformer2DModel
    except ImportError:
        return {"skipped": "Diffusers 875373ac not importable"}
    sys.path.insert(0, str(test_dir))
    import test_h3_image_packed_layout as t  # noqa: E402

    result: dict[str, Any] = {}
    for dtype in (torch.float64, torch.float32):
        torch.manual_seed(0)
        model = MiniMaxH3ImageTransformer2DModel(**t.TINY).to("cuda", dtype).eval()
        layout = t._layout()
        generator = torch.Generator().manual_seed(1)
        image = torch.randn(1, layout.image_indices.numel(), 16, generator=generator)
        text = torch.randn(1, layout.num_text_tokens, 8, generator=generator)
        image, text = image.to("cuda", dtype), text.to("cuda", dtype)
        steps = (layout.source_kind == pl.SOURCE_REFERENCE).long()

        def forward(lay: pl.PackedLayout, step_idx: torch.Tensor) -> torch.Tensor:
            return model(
                hidden_states=image,
                encoder_hidden_states=text,
                timestep=torch.tensor([0.25, 0.999], dtype=dtype, device="cuda"),
                timestep_indices=step_idx.cuda(),
                token_tags=lay.token_tags.cuda(),
                position_ids=lay.position_ids.cuda(),
                image_indices=lay.image_indices.cuda(),
                text_indices=lay.text_indices.cuda(),
                pdd_step_index=3,
                return_dict=False,
            )[0]

        with torch.no_grad():
            base = forward(layout, steps)
            again = forward(layout, steps)
            drifts, bitwise = [], 0
            for seed in range(20):
                order = torch.randperm(
                    layout.sequence_length, generator=torch.Generator().manual_seed(seed)
                )
                out = forward(pl.permute(layout, order), steps[order])
                diff = float((out - base).abs().max())
                drifts.append(diff)
                bitwise += diff == 0.0
        result[str(dtype).split(".")[-1]] = {
            "repeat_bitwise_equal": bool(torch.equal(base, again)),
            "permutations": 20,
            "bitwise_equal": bitwise,
            "max_abs": max(drifts),
        }
    return result


def collect(test_dir: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "environment": environment(),
        "fingerprint": fp.build_fingerprint()["fingerprint_sha256"],
        "position_grid": position_grid(),
        "canvas": canvas(),
        "packed_layout": packed_layout(),
        "patch_pack": patch_pack(),
        "trajectory": trajectory(test_dir),
        "permutation_drift": permutation_drift(test_dir),
    }


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, item, out)
    else:
        out[prefix] = value


def compare(ours: dict[str, Any], ref: dict[str, Any]) -> int:
    a, b = {}, {}
    _flatten("", ours, a)
    _flatten("", ref, b)
    hard, pillow = [], []
    for key in sorted(set(a) | set(b)):
        if key.startswith(REPORT_ONLY):
            continue
        if key.split(".")[-1].startswith("cuda") and (key not in a or key not in b):
            continue  # one side had no CUDA
        if a.get(key) != b.get(key):
            pillow_key = key in PILLOW_DEPENDENT or key.startswith(
                tuple(f"{prefix}." for prefix in PILLOW_DEPENDENT)
            )
            (pillow if pillow_key else hard).append(key)
    print(f"compared against reference ({ref['environment'].get('device_name', 'cpu')})")
    print(f"hardware-independent digests differing: {len(hard)}")
    for key in hard:
        print(f"  {key}: {a.get(key)} != {b.get(key)}")
    print(
        f"Pillow-dependent resize digests differing: {len(pillow)} "
        f"(pillow {ours['environment']['pillow']} vs {ref['environment']['pillow']})"
    )
    return 1 if hard else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument(
        "--tests",
        type=Path,
        default=Path("tests/models/minimax_h3_image"),
        help="the checkout's tests/models/minimax_h3_image (run from the checkout root)",
    )
    args = parser.parse_args()
    result = collect(args.tests.resolve())
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out.name}")
    print(json.dumps(result["permutation_drift"], indent=1))
    if args.compare:
        return compare(result, json.loads(args.compare.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
