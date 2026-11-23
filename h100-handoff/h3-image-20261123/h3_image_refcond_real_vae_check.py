#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Integrated-head check: image_vae_contract + image_reference_conditioning vs Diffusers.

Real released VAE (vae_diffusers/ of MiniMax-H3-Image 8ced6006), three references of
heterogeneous size, one 64x64-latent target. Our arm: ImageVAEContract.encode_references
(posterior draws recorded into the DrawLedger) -> draw_target_noise -> condition_references.
Reference arm: the pinned Diffusers EditVaeEncoderStep -> PrepareLatentsStep ->
EditPrepareLatentsStep run as they are, then MiniMaxH3ImageLoopSchedulerStep for six steps.

    PYTHONPATH=<integrated checkout> python h3_image_refcond_real_vae_check.py \
        --vae-dir <MiniMax-H3-Image>/vae_diffusers --out check.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from rl_engine.models.minimax_h3.trajectory import tensor_digest
from rl_engine.models.minimax_h3_image import image_vae as iv
from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import patch_pack as pp
from rl_engine.models.minimax_h3_image import reference_conditioning as rc

IMAGE_HW = [(520, 392), (300, 700), (1030, 1030)]
TARGET = (64, 64)
SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724]
SIGMAS += [0.23200887441635132, 0.0]


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().cpu().view(torch.int32), b.contiguous().cpu().view(torch.int32)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(iv.__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "tools/validation/models"))
    import h3_image_vae_evidence as ev  # noqa: E402
    from diffusers import AutoencoderKLMiniMaxH3Image
    from diffusers.modular_pipelines.minimax_h3_image import before_denoise as bd
    from diffusers.modular_pipelines.minimax_h3_image import denoise as dn
    from diffusers.modular_pipelines.minimax_h3_image import encoders as enc
    from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import (
        MiniMaxH3ImageModularPipeline as P,
    )
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    vae = iv.load_release_vae(AutoencoderKLMiniMaxH3Image, args.vae_dir, "cuda")
    contract = iv.ImageVAEContract(vae)
    refs = [ev.synthetic_image(h, w, seed=20 + k) for k, (h, w) in enumerate(IMAGE_HW)]

    def components():
        c = ev.release_components(vae, "cuda")
        c.reference_noise_aug = P.reference_noise_aug.fget(c)
        c.scheduler = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
        return c

    def theirs(gen_kind: str, pinned: bool):
        g = ev._gen(gen_kind, 99)
        with ev._flags(pinned):
            c = components()
            state = PipelineState()
            state.set("reference_images", [ev._to_pil(i) for i in refs])
            state.set("generator", g)
            c, state = enc.MiniMaxH3ImageEditVaeEncoderStep()(c, state)
            lat = state.get("reference_latents")
            num_rows = sum((z.shape[-2] // 2) * (z.shape[-1] // 2) for z in lat)
            state.set("latent_height", TARGET[0])
            state.set("latent_width", TARGET[1])
            c, state = bd.MiniMaxH3ImagePrepareLatentsStep()(c, state)
            state.set("num_reference_rows", num_rows)
            c, state = bd.MiniMaxH3ImageEditPrepareLatentsStep()(c, state)
        return c, state.get("latents"), num_rows, g, [tuple(z.shape[-2:]) for z in lat]

    def ours(gen_kind: str):
        g = ev._gen(gen_kind, 99)
        ledger = rc.DrawLedger(g)
        outs = contract.encode_references(refs, g)
        for i, o in enumerate(outs):
            (draw,) = o.record["generator"]["draws"]
            shape = (1, 24, *o.record["geometry"]["latent"])
            ledger.record("posterior", i, shape, draw["before"], draw["after"])
        target = rc.draw_target_noise(ledger, TARGET, "cuda")
        grids = [tuple(o.tensor.shape[-2:]) for o in outs]
        tags = torch.ones(16, dtype=torch.long)
        tags[1:9] = pl.IMAGE_TAG
        layout = pl.build_layout(tags, grids, TARGET)
        cond = rc.condition_references(
            [o.tensor for o in outs], ledger=ledger, device="cuda", layout=layout
        )
        ledger.check_order(len(refs))
        return cond, pp.ImagePatchPacker().pack_image(target)[0], ledger, g

    def loop(c, cond, image_rows):
        sched = c.scheduler
        t = torch.tensor([1.0 - s for s in SIGMAS[:-1]], dtype=torch.float32)
        sched.set_timesteps(sigmas=torch.cat([1.0 - t, torch.zeros(1)]), device="cuda")
        state = SimpleNamespace(
            latents=image_rows, noise_pred=None, num_reference_rows=cond.num_reference_rows
        )
        velocity = torch.Generator().manual_seed(5)
        digests, target0 = [], tensor_digest(image_rows[cond.num_reference_rows :])
        for i, ti in enumerate(sched.timesteps):
            state.noise_pred = torch.randn(1, *image_rows.shape, generator=velocity).cuda()
            dn.MiniMaxH3ImageLoopSchedulerStep()(c, state, i, ti)
            digests.append(cond.check_fixed(state.latents, step=i))
        moved = tensor_digest(state.latents[cond.num_reference_rows :]) != target0
        return {"steps": len(digests), "fixed": set(digests) == {cond.digests["rows"]}, "target_moved": moved}

    results: dict = {}
    ok = True
    for pinned in (False, True):
        for gen_kind in ("cpu", "cuda"):
            c, image_rows, num_rows, g_ref, grids = theirs(gen_kind, pinned)
            cond, target_rows, ledger, g_ours = ours(gen_kind)
            row = {
                "reference_grids": [list(x) for x in grids],
                "num_reference_rows": num_rows,
                "reference_rows_bitwise_equal": _bits_equal(image_rows[:num_rows], cond.rows),
                "target_rows_bitwise_equal": _bits_equal(image_rows[num_rows:], target_rows),
                "generator_state_equal": bool(g_ref.get_state().equal(g_ours.get_state())),
                "draw_order": [[d.stage, d.index] for d in ledger.draws],
                "rows_sha256": cond.digests["rows"],
                "loop": loop(c, cond, image_rows),
            }
            ok &= all(
                row[k]
                for k in (
                    "reference_rows_bitwise_equal",
                    "target_rows_bitwise_equal",
                    "generator_state_equal",
                )
            ) and row["loop"]["fixed"] and row["loop"]["target_moved"]
            results[f"{'pinned' if pinned else 'default'}_flags-{gen_kind}_generator"] = row
    out = {
        "schema": "rlkernel.h3_image.reference_conditioning_real_vae_check.v1",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "diffusers": __import__("diffusers").__version__,
        },
        "images": [list(x) for x in IMAGE_HW],
        "target_latent": list(TARGET),
        "arms": results,
        "all_green": ok,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for k, v in results.items():
        print(k, {kk: vv for kk, vv in v.items() if kk not in ("draw_order", "rows_sha256")})
    print("all_green", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
