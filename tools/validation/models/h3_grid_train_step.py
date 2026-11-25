# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""G5 of RFC #420 ``final_distributed_h3_run``: LoRA training steps on the TP x CP grid equal the
single-GPU run bit for bit (``h3-grid-train-v1``).

Every rank runs the same schedule twice, one after the other:

1. **canonical**: the full model on ``Grid(1, 1)``, the LoRA whole;
2. **grid**: the weight-sharded model on the TP x CP grid, this rank's LoRA shards.

The schedule is ``--optimizer-steps`` optimizer steps of ``--passes`` forward + backward passes
each (micro-batches x SDE steps in a trainer), every pass with its own inputs, timesteps and
output cotangent, the loss divided by the number of passes; then ``finish_grads``, the exact clip
to ``--max-norm`` and one AdamW step (the recipe's lr 3e-4, weight decay 1e-4). After each
optimizer step the canonical run records the gradient norm and the SHA-256 of this rank's cut of
every accumulated gradient (after the refiner broadcast, before the clip) and of every updated LoRA
tensor; the grid run must match them all. ``--repeat`` runs the grid schedule a
second time from the same initial LoRA, which is where a non-repeatable backward (the provider
refiner on B200) would show. Each rank compares with its own single-GPU run: the grid's refiner
gradients are global rank 0's (``finish_grads``), so ``refiner_bitwise`` on another rank also asks
that rank's refiner backward to equal rank 0's. That is stricter than the training step needs, and
can only fail a run, not pass one. The grid run also exports its LoRA as the rollout reads it and
checks that both CP replicas of each TP rank hold the same tensors.

    srun --ntasks 8 python tools/validation/models/h3_grid_train_step.py \\
        --checkpoint <dir with transformer/> --tp 4 --cp 2 --sp --cp-mode ring --size 925 \\
        --provider-boundaries --refiner-lora --repeat --out <dir>

``--pp P`` runs the grid as ``P`` pipeline stages (``h3-pipeline-v1``; ``P * tp * cp`` ranks): each
rank compares the tensors its stage holds, rank 0 checks that the ranks together hold every tensor
of the canonical run, the passes of an optimizer step go through ``--schedule`` (the repeat through
``--repeat-schedule``, so it also shows that the schedule changes no byte), and the exported files,
written by the first stage, must hold the canonical run's final LoRA.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl
from rl_engine.integrations.engines.rollout.vllm.h3_omni_model import tensor_digest
from rl_engine.integrations.engines.train import h3_grid_actor as ga
from rl_engine.validation.models import h3_lora
from rl_engine.validation.models import h3_pipeline as pipe
from rl_engine.validation.models import h3_sharded_load as sl
from rl_engine.validation.models import h3_tp

SIZES = {
    "40": dict(text=8, frames=1, side=8, audio=8),
    "925": dict(text=77, frames=3, side=32, audio=40),
    "928": dict(text=80, frames=3, side=32, audio=80),
    # about the recipe's 256 x 384, 121-frame request (4296 rows, 1024 text tokens)
    "recipe": dict(text=1024, frames=31, side=20, audio=86),
}


def _inputs(size: str, step: int, n: int) -> dict:
    """Pass ``n`` of optimizer step ``step``: its own latents, text rows and two timesteps."""

    from tools.validation.models.h3_runtime_report import sized_inputs

    base = sized_inputs("cuda", **SIZES[size])
    gen = torch.Generator().manual_seed(7919 * step + 104729 * n + 17)
    for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
        base[key] = torch.randn(base[key].shape, generator=gen).to("cuda", base[key].dtype)
    t = torch.tensor([37.0 * (n + 1) + 101.0 * step, 500.0 + 13.0 * n], device="cuda")
    base["timestep"] = t
    return base


def _loss_fn(step: int, n: int, passes: int):
    """``(<video, Gv> + <audio, Ga>) / passes`` with seeded FP32 cotangents."""

    def loss(video, audio):
        gen = torch.Generator().manual_seed(15485863 * step + 32452843 * n + 3)
        gv = torch.randn(video.shape, generator=gen).to(video.device)
        ga_ = torch.randn(audio.shape, generator=gen).to(audio.device)
        return ((video.float() * gv).sum() + (audio.float() * ga_).sum()) / passes

    return loss


def _cut(name: str, t: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """This TP rank's cut of the canonical (whole) LoRA tensor ``name``."""

    if name.startswith("refiner."):
        return t  # held whole on every rank
    _, target, part = _split(name)
    kind = pl.TARGETS[target]
    if part == "A":
        return par.shard_cols(t, tp, rank) if kind == "row" else t
    return t if kind == "row" else par.shard_rows(t, tp, rank)


def _split(name: str) -> tuple[str, str, str]:
    """``<block>.<target>.<part>``; targets (``attn.to_out.0``) and parts (``B.gate``) hold dots."""

    block, rest = name.split(".", 1)
    for target in sorted(pl.TARGETS, key=len, reverse=True):
        if rest.startswith(target + "."):
            return block, target, rest[len(target) + 1 :]
    raise ValueError(f"no LoRA target in {name!r}")


def _prepare(model) -> None:
    model.enable_gradient_checkpointing()
    for p in model.parameters():
        p.requires_grad_(False)


def _schedule(actor, opt, args, cut, inject=None, schedule="serial") -> list[dict]:
    """Run the schedule; after each optimizer step, the norm and ``cut(name, tensor)``'s digests.

    On a pipeline the passes of one optimizer step are interleaved by ``schedule``.

    ``inject = (name, rank)``: before the first clip, global rank ``rank`` changes the first element
    of tensor ``name``'s accumulated gradient by about one BF16 step (a negative control; Adam's
    first step is ``lr * sign(g)``, so the gradient digests, not the parameters, must show it).
    """

    steps = []
    for step in range(args.optimizer_steps):
        began = time.perf_counter()
        results, _ = actor.forward_backward_many(
            [_inputs(args.size, step, n) for n in range(args.passes)],
            lambda n, video, audio: _loss_fn(step, n, args.passes)(video, audio),
            schedule=schedule,
        )
        losses = [float(loss) for loss in results]
        if inject is not None and step == 0 and actor.rank == inject[1]:
            g = actor.parameters()[actor.names.index(inject[0])].grad.view(-1)
            g[0] = g[0] * 1.0078125 + 2.0**-20
        actor.finish_grads()
        grads = {
            name: tensor_digest(cut(name, p.grad))
            for name, p in zip(actor.names, actor.parameters())
            if p.grad is not None
        }
        norm = actor.clip_grad_norm_(args.max_norm)
        if norm == norm and norm not in (float("inf"), float("-inf")):
            opt.step()
        actor.zero_grad()
        torch.cuda.synchronize()
        digests = {
            name: tensor_digest(cut(name, p.detach()))
            for name, p in zip(actor.names, actor.parameters())
        }
        steps.append(
            {
                "norm": norm,
                "losses": losses,
                "seconds": time.perf_counter() - began,
                "grads": grads,
                "digests": digests,
            }
        )
    return steps


def _export(actor, out: Path, *, injected: bool) -> dict:
    """Export the LoRA and check the file against it.

    In an injected run the CP replicas may no longer agree (``export`` then raises on every rank):
    that is recorded, so the run still reports whether the injection was detected.
    """

    try:
        exported = actor.export(str(out / "export"))
    except RuntimeError as exc:
        if not injected:
            raise
        return {"error": str(exc), "file_digests_equal": False, "cp_replicas_agree": False}
    path = exported["path"]
    back = torch.load(path, weights_only=True)
    mine = actor.digests()
    return {
        "path": os.path.relpath(path, out),
        # a pipeline's file holds every stage's tensors; without one it holds exactly this rank's
        "file_digests_equal": back["digests"] == exported["digests"]
        and all(back["digests"].get(k) == v for k, v in mine.items())
        and (actor.pipeline is not None or len(mine) == len(back["digests"])),
        "file_digests": back["digests"],
        "cp_replicas_agree": True,  # export raises otherwise
    }


def _owns(stage, name: str) -> bool:
    """Whether the stage (``None``: no pipeline) holds the LoRA tensor ``name``."""

    if stage is None:
        return True
    return stage.first if name.startswith("refiner.") else stage.owns_block(int(_split(name)[0]))


def _compare(reference: list[dict], run: list[dict], stage=None) -> dict:
    out = []
    for i, (whole, got) in enumerate(zip(reference, run)):
        ref = {
            **whole,
            **{
                part: {k: v for k, v in whole[part].items() if _owns(stage, k)}
                for part in ("digests", "grads")
            },
        }
        bad = sorted(k for k in ref["digests"] if ref["digests"][k] != got["digests"].get(k))
        bad_g = sorted(k for k in ref["grads"] if ref["grads"][k] != got["grads"].get(k))
        if set(ref["grads"]) != set(got["grads"]):
            bad_g = sorted(set(bad_g) | (set(ref["grads"]) ^ set(got["grads"])))
        out.append(
            {
                "step": i,
                "norm_equal": ref["norm"] == got["norm"],
                "losses_equal": ref["losses"] == got["losses"],
                "block_tensors_mismatched": [k for k in bad if not k.startswith("refiner.")][:10],
                "block_mismatches": sum(1 for k in bad if not k.startswith("refiner.")),
                "refiner_mismatches": sum(1 for k in bad if k.startswith("refiner.")),
                "block_grads_mismatched": [k for k in bad_g if not k.startswith("refiner.")][:10],
                "block_grad_mismatches": sum(1 for k in bad_g if not k.startswith("refiner.")),
                "refiner_grad_mismatches": sum(1 for k in bad_g if k.startswith("refiner.")),
                "other_tensors": sorted(set(got["digests"]) - set(ref["digests"]))[:10],
                "tensors_compared": len(ref["digests"]),
                "grads_compared": len(ref["grads"]),
            }
        )
    return {
        "steps": out,
        "blocks_bitwise": all(
            s["norm_equal"]
            and s["losses_equal"]
            and not s["block_mismatches"]
            and not s["block_grad_mismatches"]
            and not s["other_tensors"]
            for s in out
        )
        and len(run) == len(reference),
        "refiner_bitwise": all(
            not s["refiner_mismatches"] and not s["refiner_grad_mismatches"] for s in out
        ),
    }


def _check_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse arguments with which a run would prove nothing."""

    if args.optimizer_steps < 1 or args.passes < 1:
        ap.error("--optimizer-steps and --passes must be at least 1")
    if args.inject is None:
        return
    name, colon, at = args.inject.rpartition(":")
    if not colon or not at.isdecimal():
        ap.error("--inject takes NAME:RANK")
    pp = getattr(args, "pp", 1)
    if int(at) >= args.tp * args.cp * pp:
        ap.error(f"--inject: no rank {at} in a pp {pp} x tp {args.tp} x cp {args.cp} run")
    if name.startswith("refiner."):
        # finish_grads replaces every rank's refiner gradients with rank 0's
        ap.error("--inject: the refiner broadcast would overwrite it; name a block tensor")
    if pp > 1:  # every rank refuses alike
        stage = pipe.StageSpec.make(50, pp, int(at) // (args.tp * args.cp))
        if not stage.owns_block(int(_split(name)[0])):
            ap.error(f"--inject: rank {at} is on another stage than {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--sp", action="store_true")
    ap.add_argument("--cp-mode", choices=("ulysses", "ring"), default="ring")
    ap.add_argument("--size", choices=tuple(SIZES), default="925")
    ap.add_argument("--provider-boundaries", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=float, default=128.0)
    ap.add_argument("--lora-seed", type=int, default=7)
    ap.add_argument("--lora-b-zero", action="store_true")
    ap.add_argument("--refiner-lora", action="store_true")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--optimizer-steps", type=int, default=2)
    ap.add_argument("--max-norm", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--pp", type=int, default=1, help="pipeline stages")
    ap.add_argument("--schedule", choices=pipe.SCHEDULES, default="1f1b")
    ap.add_argument("--repeat-schedule", choices=pipe.SCHEDULES, default="serial")
    ap.add_argument(
        "--inject",
        help="NAME:RANK, e.g. 17.attn.to_v.A:2: perturb that gradient on that rank in the first "
        "grid run (negative control; the verdict is inject_detected, pass is then false). A block "
        "tensor only",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _check_args(ap, args)

    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '29521')}",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", local),
    )
    if world != args.tp * args.cp * args.pp:
        raise SystemExit(f"world {world} != tp {args.tp} x cp {args.cp} x pp {args.pp}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    singles = [dist.new_group([r]) for r in range(world)]  # every rank creates every group
    strict = not args.provider_boundaries
    lora_kw = dict(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        seed=args.lora_seed,
        b_zero=args.lora_b_zero,
        refiner=args.refiner_lora,
    )
    opt_kw = dict(lr=args.lr, weight_decay=args.weight_decay)
    layout = stage = None
    if args.pp > 1:
        layout = pipe.make_layout(args.pp, args.tp, args.cp, cp_mode=args.cp_mode)
        grid, stage = layout.grid, layout.stage
    else:
        grid = h3_tp.make_grid(args.tp, args.cp, attention="h3_full", cp_mode=args.cp_mode)
    transformer = args.checkpoint / "transformer"

    # 1. canonical: the whole model and LoRA on Grid(1, 1); this rank's cut of each tensor
    model = sl.load_sharded_model(transformer, 1, 0, "cuda").eval()
    _prepare(model)
    h3_lora.attach_lora(model, 1, 0, **lora_kw)
    canonical = ga.H3GridActor(
        model,
        h3_tp.Grid(1, 1, 0, 0, attention="h3_full"),
        sp=False,
        strict_boundaries=strict,
        group=singles[rank],
    )
    opt = ga.make_optimizer(canonical.parameters(), **opt_kw)
    reference = _schedule(
        canonical, opt, args, lambda name, t: _cut(name, t, args.tp, grid.tp_rank)
    )
    del canonical, opt, model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. the grid, once (and again with --repeat), from the same initial LoRA
    model = sl.load_sharded_model(transformer, grid.tp, grid.tp_rank, "cuda", stage=stage).eval()
    _prepare(model)
    runs, exported = [], None
    rows = int(_inputs(args.size, 0, 0)["token_tags"].numel())  # before the peak is measured
    torch.cuda.reset_peak_memory_stats()
    for schedule in (args.schedule, args.repeat_schedule)[: 2 if args.repeat else 1]:
        h3_lora.attach_lora(model, grid.tp, grid.tp_rank, stage=stage, **lora_kw)
        actor = ga.H3GridActor(model, grid, sp=args.sp, strict_boundaries=strict, pipeline=layout)
        opt = ga.make_optimizer(actor.parameters(), **opt_kw)
        inject = None
        if args.inject and not runs:
            name, at = args.inject.rsplit(":", 1)
            if name not in actor.names:
                raise SystemExit(f"--inject: no LoRA tensor {name!r}")
            inject = (name, int(at))
        runs.append(_schedule(actor, opt, args, lambda name, t: t, inject, schedule))
        if exported is None:
            exported = _export(actor, args.out, injected=inject is not None)
        readback = actor.readback()
        del actor, opt

    report = {
        "contract": ga.CONTRACT,
        "rank": rank,
        "tp": args.tp,
        "cp": args.cp,
        "sp": args.sp,
        "cp_mode": args.cp_mode,
        "size": args.size,
        "rows": rows,
        "strict_boundaries": strict,
        "refiner_lora": args.refiner_lora,
        "passes": args.passes,
        "optimizer_steps": args.optimizer_steps,
        "max_norm": args.max_norm,
        "norms": [s["norm"] for s in reference],
        "pp": args.pp,
        "stage": None if stage is None else vars(stage),
        "schedules": [args.schedule, args.repeat_schedule] if args.pp > 1 else None,
        "grid_vs_canonical": _compare(reference, runs[0], stage),
        "tensor_names": sorted(runs[0][0]["digests"]),
        "canonical_tensors": len(reference[0]["digests"]),
        "canonical_names": sorted(reference[0]["digests"]) if rank == 0 else None,
        # the file the rollout would load holds this TP rank's cut of the canonical final LoRA
        "export_is_canonical": exported.pop("file_digests", None) == reference[-1]["digests"],
        "grid_repeat": _compare(runs[0], runs[1]) if args.repeat else None,
        "export": exported,
        "grid_seconds_per_step": [s["seconds"] for s in runs[0]],
        "canonical_seconds_per_step": [s["seconds"] for s in reference],
        "grid_peak_bytes": torch.cuda.max_memory_allocated(),
        "readback": readback,
    }
    if args.inject:
        name, at = args.inject.rsplit(":", 1)
        first = report["grid_vs_canonical"]["steps"][0]
        report["inject"] = {
            "tensor": name,
            "rank": int(at),
            "reported_here": name in first["block_grads_mismatched"],
            "mismatches_here": first["block_grad_mismatches"],
        }
    # an infinite or NaN norm skips the optimizer step on both sides, so equal digests would
    # then prove nothing: such a run fails
    report["norms_finite"] = all(math.isfinite(n) for n in report["norms"])
    report["pass"] = bool(
        report["norms_finite"]
        and report["grid_vs_canonical"]["blocks_bitwise"]
        and (not args.refiner_lora or report["grid_vs_canonical"]["refiner_bitwise"])
        and (not args.repeat or report["grid_repeat"]["blocks_bitwise"])
        and (not (args.repeat and args.refiner_lora) or report["grid_repeat"]["refiner_bitwise"])
        and exported["file_digests_equal"]
        and (args.pp == 1 or report["export_is_canonical"])
    )
    everyone = [None] * world
    dist.all_gather_object(everyone, report)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        names = everyone[0].pop("canonical_names")
        held = set().union(*(r["tensor_names"] for r in everyone))
        for r in everyone:
            del r["tensor_names"]
            r.pop("canonical_names", None)
        summary = {
            "pass": all(r["pass"] for r in everyone) and held == set(names),
            "every_canonical_tensor_is_held": held == set(names),
            "ranks": [
                {
                    "rank": r["rank"],
                    "pass": r["pass"],
                    "blocks_bitwise": r["grid_vs_canonical"]["blocks_bitwise"],
                    "refiner_bitwise": r["grid_vs_canonical"]["refiner_bitwise"],
                    "repeat_blocks_bitwise": (r["grid_repeat"] or {}).get("blocks_bitwise"),
                    "repeat_refiner_bitwise": (r["grid_repeat"] or {}).get("refiner_bitwise"),
                }
                for r in everyone
            ],
            "norms": everyone[0]["norms"],
        }
        if args.inject:
            injected = [r for r in everyone if r["rank"] == r["inject"]["rank"]]
            others = [r for r in everyone if r["rank"] != r["inject"]["rank"]]
            summary["inject_detected"] = bool(
                injected
                and injected[0]["inject"]["reported_here"]
                and injected[0]["inject"]["mismatches_here"] == 1
                and all(r["inject"]["mismatches_here"] == 0 for r in others)
            )
        (args.out / "report.json").write_text(
            json.dumps({"summary": summary, "ranks": everyone}, indent=1)
        )
        print(json.dumps(summary))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
