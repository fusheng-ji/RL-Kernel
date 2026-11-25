# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Full 50-block H3 model on a TP x CP (x SP) grid against its single-GPU canonical run.

Launch one process per GPU, for example on one 8-GPU node::

    srun -N1 --ntasks=8 python tools/validation/models/h3_parallel_full_model.py \\
        --checkpoint <dir with transformer/> --tp 4 --cp 2 --sp --cp-mode ring --size 928 \\
        --out evidence/ws2_full_tp4_cp2_sp

Every rank loads the real checkpoint, runs the model with every block on the canonical
``Grid(1, 1, 0, 0)`` (``h3_parallel_model.parallel_model_execution``), then on the requested grid,
and compares, bit for bit: the video / audio velocity, the gradients of the three inputs, every
block's output and output gradient (first drift with owning rank and logical row, if any), and
this rank's weight-shard gradients of all 350 sharded projections (device checksums). With
``--inject BLOCK:ROW:RANK`` that rank perturbs one row of the block's output: a negative control
that must be reported at exactly that block and row with that owner. The rank must hold the row;
without SP, where the TP ranks of a CP rank hold the same rows, it must be the last of them, whose
copy the assembled output keeps (anything else is refused).

With ``--sharded`` the grid run uses a model loaded by ``h3_sharded_load.load_sharded_model``:
after the canonical run the full model is freed, and each rank loads only its shards of the
sharded projections (the rest of the model in full), so the comparison above holds for a grid
run that never holds those full weights. The canonical run before it still loads the full model
on every GPU, so each GPU needs the full model's memory for that phase. ``--provider-boundaries``
leaves the boundary modules on the provider in both runs, for GPUs the strict boundary rows do not
support (they need SM90).

``--lora-rank R`` puts a seeded LoRA (``h3_lora.make_block_lora``: A normal with std 1/R, B
random or, with ``--lora-b-zero``, zero; scale ``--lora-alpha / R``) on the six targets of every
block, cut for each run's grid (``h3-parallel-lora-v2``), and compares every LoRA shard's
gradient too (``lora_gradients_bitwise``). The two token-refiner blocks are not on the grid and
carry no LoRA
here.

``--pp P`` runs the grid as ``P`` pipeline stages of ``tp * cp`` ranks (``h3-pipeline-v1``,
docs/operators/h3-parallel.md): each rank compares what its stage computed (with ``--sharded`` it
holds only that), and rank 0 merges the drift reports and checks that the stages hold every block
once. ``--inject-handoff BLOCK:ROW`` perturbs that row as it enters the stage that starts at
``BLOCK`` (``:grad``: the gradient going back; ``:wire``: before the digest, so the run must be
refused). ``--train-time-embedder`` also compares the time embedder's gradients, the one FP32 chain
that runs through every stage (TP only: under CP / SP each rank holds a partial one). With
``--provider-boundaries`` this also turns on PyTorch's deterministic kernels in both runs: the
provider's final AdaLN gathers rows, and that gather's default CUDA backward does not repeat, so
the single-GPU run would differ from itself.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl
from rl_engine.validation.models import h3_block as hb
from rl_engine.validation.models import h3_lora
from rl_engine.validation.models import h3_parallel_model as pm
from rl_engine.validation.models import h3_pipeline as pipe
from rl_engine.validation.models import h3_sharded_load as sl
from rl_engine.validation.models import h3_tp
from rl_engine.validation.models.h3_full_model import FullModelTrace

# Layout name -> sized_inputs arguments (None: the 12-token layout of the full-model harness).
SIZES = {
    "12": None,
    "40": dict(text=8, frames=1, side=8, audio=8),
    "925": dict(text=77, frames=3, side=32, audio=40),
    "928": dict(text=80, frames=3, side=32, audio=80),
}
INPUTS = ("hidden_states", "audio_hidden_states", "encoder_hidden_states")


def _inputs(size: str):
    """The packed request of ``size`` on CUDA."""
    from tools.validation.models.h3_full_model import packed_inputs
    from tools.validation.models.h3_runtime_report import sized_inputs

    return packed_inputs("cuda") if SIZES[size] is None else sized_inputs("cuda", **SIZES[size])


def _run(model, base, grid, sp, ops, gradients, digest_leaf, *, strict, inject=None):
    """Forward + backward on ``grid``: outputs, input gradients, snapshots, shard-gradient digests.

    ``digest_leaf(name, grad)`` reduces each weight-shard gradient to its digest as it is
    accumulated (the run never holds more than one block's worth of them).
    """

    inputs = {
        k: (v.detach().clone().requires_grad_() if k in INPUTS else v) for k, v in base.items()
    }
    trace = FullModelTrace()
    trace.capture = True
    began = time.perf_counter()
    with pm.parallel_model_execution(
        model, grid, sp=sp, trace=trace, ops=ops, strict_boundaries=strict
    ) as state:
        state.inject = inject
        state.digest_leaf = digest_leaf
        video, audio = model(**inputs)
        torch.autograd.backward([video, audio], list(gradients))
        torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    return {
        "video": video.detach().cpu(),
        "audio": audio.detach().cpu(),
        "input_grads": {k: inputs[k].grad.detach().cpu() for k in INPUTS},
        "snapshots": dict(trace.snapshots),
        "leaf_digests": state.digests,
        "seconds": seconds,
    }


def _run_pipeline(
    model, base, layout, sp, ops, gradients, digest_leaf, *, strict, inject=None, corrupt=None
):
    """``_run`` on this rank's pipeline stage. ``inject`` names a global rank.

    ``refused`` is the error of a handoff that did not verify (the outputs are then missing).
    """

    inputs = {
        k: (v.detach().clone().requires_grad_() if k in INPUTS else v) for k, v in base.items()
    }
    trace = FullModelTrace()
    trace.capture = True
    runner = pipe.Pipeline(model, layout, sp=sp, strict_boundaries=strict, ops=ops)
    runner.corrupt = corrupt

    def configure(state):
        state.digest_leaf = digest_leaf
        if inject is not None and layout.stage.owns_block(inject[0]):
            state.inject = (inject[0], inject[1], inject[2] - layout.grid.first_rank)

    began, refused, outputs = time.perf_counter(), None, [(None, None)]
    states = []
    try:
        outputs, _ = runner.run(
            [inputs],
            lambda _, video, audio: pipe.SeededCotangent.apply(video, audio, *gradients),
            trace=trace,
            configure=lambda state: (states.append(state), configure(state)),
        )
    except pipe.HandoffError as exc:
        refused = str(exc)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    video, audio = outputs[0]
    return {
        "video": None if video is None else video.cpu(),
        "audio": None if audio is None else audio.cpu(),
        "input_grads": {
            k: inputs[k].grad.detach().cpu() for k in INPUTS if inputs[k].grad is not None
        },
        "snapshots": dict(trace.snapshots),
        "leaf_digests": states[0].digests,
        "seconds": seconds,
        "refused": refused,
        "readback": runner.readback(),
        "packed_bindings": dict(trace.packed_bindings),
        "boundary_bindings": dict(trace.boundary_bindings),
    }


def _time_embedder_grads(model) -> dict[str, torch.Tensor]:
    """The time embedder's gradients on the CPU, cleared on the model."""

    out = {}
    for name, p in model.time_embedder.named_parameters():
        if p.grad is not None:
            out[name], p.grad = p.grad.detach().cpu(), None
    return out


def parse_handoff(text: str, stage_starts: list[int], rows: int):
    """``BLOCK:ROW[:grad][:wire]`` as a ``Corruption``; a block that starts no later stage is
    refused."""

    parts = text.split(":")
    flags = parts[2:]
    if len(parts) < 2 or set(flags) - {"grad", "wire"} or len(set(flags)) != len(flags):
        raise ValueError(f"--inject-handoff takes BLOCK:ROW[:grad][:wire], got {text!r}")
    block, row = int(parts[0]), int(parts[1])
    if block not in stage_starts[1:]:
        raise ValueError(f"block {block} starts no later stage (they start at {stage_starts[1:]})")
    if not 0 <= row < rows:
        raise ValueError(f"row {row} is not in the {rows} packed rows")
    return pipe.Corruption(block, row, wire="wire" in flags, grad="grad" in flags)


def shard_gradient_verdict(expected: dict, observed: dict, blocks: int, *, lora: bool) -> dict:
    """Compare the canonical run's shard-gradient digests with the grid's.

    Every block's every sharded projection (and, with ``lora``, every LoRA shard) must be in
    both runs with the same digest: an empty or partial comparison is not a pass, and a key that
    only one run holds is reported as missing. ``base_compared`` / ``lora_compared`` count the
    keys both runs hold (the ones whose digests were actually compared).
    """

    missing = sorted(set(expected) ^ set(observed))
    mismatched = sorted(k for k in set(expected) & set(observed) if expected[k] != observed[k])
    lora_keys = sorted(k for k in expected if ".lora." in k)
    base_keys = [k for k in expected if ".lora." not in k]
    base_mismatched = [k for k in mismatched if ".lora." not in k]
    lora_mismatched = [k for k in mismatched if ".lora." in k]
    base_missing = [k for k in missing if ".lora." not in k]
    lora_missing = [k for k in missing if ".lora." in k]
    both = set(expected) & set(observed)
    return {
        "missing": missing,
        "base_compared": sum(1 for k in both if ".lora." not in k),
        "lora_compared": sum(1 for k in both if ".lora." in k),
        "base_mismatched": base_mismatched,
        "lora_mismatched": lora_mismatched,
        "lora_keys": lora_keys,
        "base_bitwise": not base_mismatched
        and not base_missing
        and len(base_keys) == blocks * len(pm.SHARDED_WEIGHTS),
        "lora_bitwise": lora
        and not lora_mismatched
        and not lora_missing
        and len(lora_keys) == blocks * h3_lora.LEAVES_PER_BLOCK,
    }


def _prepare(model, *, time_embedder: bool = False) -> None:
    """Checkpoint the blocks and freeze the parameters: the shard leaves carry the gradients."""
    model.enable_gradient_checkpointing()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.time_embedder.parameters():
        p.requires_grad_(time_embedder)


def _resident_bytes(model) -> int:
    """Bytes of the parameters and stored projection shards actually on the device."""
    total = sum(p.numel() * p.element_size() for p in model.parameters() if not p.is_meta)
    for block in model.transformer_blocks:
        stored = getattr(block, pm.SHARDS_ATTR, None)
        if stored is not None:
            total += sum(t.numel() * t.element_size() for t in stored["tensors"].values())
    return total


def main() -> None:
    """Run the canonical model and the grid on this rank, compare them, write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--pp", type=int, default=1, help="pipeline stages")
    parser.add_argument("--pp-sizes", default="", help="blocks per stage, e.g. 10,40")
    parser.add_argument(
        "--inject-handoff", default="", help="BLOCK:ROW[:grad][:wire] negative control (--pp)"
    )
    parser.add_argument(
        "--train-time-embedder", action="store_true", help="compare its gradients too (TP only)"
    )
    parser.add_argument("--attention", choices=("interim", "h3_full"), default="h3_full")
    parser.add_argument("--cp-mode", choices=("ulysses", "ring"), default="ring")
    parser.add_argument("--size", choices=tuple(SIZES), default="40")
    parser.add_argument("--local-size", type=int, default=8, help="GPUs per node")
    parser.add_argument("--inject", default="", help="BLOCK:ROW:RANK negative control")
    parser.add_argument(
        "--sharded", action="store_true", help="run the grid on a model holding only its shards"
    )
    parser.add_argument(
        "--provider-boundaries", action="store_true", help="boundary modules on the provider"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=0, help="seeded LoRA on the six targets (0: none)"
    )
    parser.add_argument("--lora-alpha", type=float, default=128.0)
    parser.add_argument("--lora-seed", type=int, default=0)
    parser.add_argument("--lora-b-zero", action="store_true", help="B = 0, the recipe's init")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
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
    if args.train_time_embedder and (args.cp > 1 or args.sp):
        raise SystemExit("--train-time-embedder: under CP / SP a rank's gradient is partial")
    if args.pp == 1 and (args.inject_handoff or args.pp_sizes):
        raise SystemExit("--inject-handoff / --pp-sizes need --pp > 1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from diffusers import MiniMaxH3Transformer3DModel

    started = time.perf_counter()
    model = (
        MiniMaxH3Transformer3DModel.from_pretrained(
            str(args.checkpoint / "transformer"),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    _prepare(model, time_embedder=args.train_time_embedder)
    load_seconds = time.perf_counter() - started
    strict = not args.provider_boundaries
    deterministic = args.train_time_embedder and not strict
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    lora_kw = dict(
        rank=args.lora_rank, alpha=args.lora_alpha, seed=args.lora_seed, b_zero=args.lora_b_zero
    )
    lora_bytes = {}
    if args.lora_rank:
        lora_bytes["canonical"] = h3_lora.attach_lora(model, 1, 0, **lora_kw)

    base = _inputs(args.size)
    tags = base["token_tags"].cpu()
    length = int(tags.numel())
    gen = torch.Generator().manual_seed(99)
    ops = hb.CandidateOps()

    def grad_like(shape):
        return torch.randn(shape, generator=gen).bfloat16().cuda()

    with torch.no_grad():
        probe = model(**{k: (v.detach() if k in INPUTS else v) for k, v in base.items()})
    gradients = [grad_like(probe[0].shape), grad_like(probe[1].shape)]
    del probe
    torch.cuda.empty_cache()

    spans = pm.rank_spans(args.tp, args.cp, length, args.sp)
    grid_kw = dict(
        local_size=args.local_size if args.tp > args.local_size else None,
        attention=args.attention,
        cp_mode=args.cp_mode,
    )
    sizes = [int(s) for s in args.pp_sizes.split(",")] if args.pp_sizes else None
    layout = corrupt = None
    if args.pp > 1:
        layout = pipe.make_layout(args.pp, args.tp, args.cp, sizes=sizes, **grid_kw)
        grid = layout.grid
        starts = [pipe.StageSpec.make(50, args.pp, s, sizes).lo for s in range(args.pp)]
        if args.inject_handoff:
            corrupt = parse_handoff(args.inject_handoff, starts, length)
    else:
        grid = h3_tp.make_grid(args.tp, args.cp, **grid_kw)
    canonical = h3_tp.Grid(1, 1, 0, 0, attention=args.attention)
    # The canonical run reduces each unsharded gradient to the digest of this rank's shard of it.
    reference = _run(
        model,
        base,
        canonical,
        False,
        ops,
        gradients,
        lambda name, grad: pm.exact_digest(pm.shard_of(grad, name, args.tp, grid.tp_rank)),
        strict=strict,
    )
    expected = reference["leaf_digests"]
    if layout is not None:
        expected = {
            k: v for k, v in expected.items() if layout.stage.owns_block(int(k.split(".", 1)[0]))
        }
    reference["time_embedder"] = _time_embedder_grads(model)
    full_param_bytes = _resident_bytes(model)
    sharded = None
    if args.sharded:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = sl.load_sharded_model(
            args.checkpoint / "transformer",
            grid.tp,
            grid.tp_rank,
            "cuda",
            stage=None if layout is None else layout.stage,
        ).eval()
        _prepare(model, time_embedder=args.train_time_embedder)
        if args.lora_rank:  # the sharded model starts without a LoRA; give it the grid's
            h3_lora.attach_lora(model, grid.tp, grid.tp_rank, **lora_kw)
        sharded = {
            "load_seconds": time.perf_counter() - started,
            "resident_param_bytes": _resident_bytes(model),
            "full_model_param_bytes": full_param_bytes,
        }
    if args.lora_rank and not args.sharded:
        h3_lora.attach_lora(model, grid.tp, grid.tp_rank, **lora_kw)
    if args.lora_rank:
        if layout is not None:
            pipe.drop_off_stage_lora(model, layout.stage)
        lora_bytes["grid"] = sum(
            t.numel() * t.element_size()
            for block in model.transformer_blocks
            for parts in (getattr(block, pm.LORA_ATTR, None) or {"tensors": {}})["tensors"].values()
            for t in parts.values()
        )
    torch.cuda.empty_cache()
    inject = None
    if args.inject:
        block, row, victim = (int(v) for v in args.inject.split(":"))
        inject = (block, row, victim)
        if layout is not None:  # every rank refuses alike, before any collective
            first = pipe.StageSpec.make(50, args.pp, 0, sizes)
            owner = next(
                s
                for s in (pipe.StageSpec.make(50, args.pp, r, sizes) for r in range(first.pp))
                if s.owns_block(block)
            )
            pm.check_injection(spans, row, victim - owner.pp_rank * layout.stage_size)
    reports: dict = {}
    if layout is None:
        observed = _run(
            model,
            base,
            grid,
            args.sp,
            ops,
            gradients,
            lambda name, grad: pm.exact_digest(grad),
            strict=strict,
            inject=inject,
        )
    else:
        observed = _run_pipeline(
            model,
            base,
            layout,
            args.sp,
            ops,
            gradients,
            lambda name, grad: pm.exact_digest(grad),
            strict=strict,
            inject=inject,
            corrupt=corrupt,
        )
        reports = {k: observed[k] for k in ("refused", "packed_bindings", "boundary_bindings")}
    if sharded is not None:
        sharded["parallel_peak_bytes"] = torch.cuda.max_memory_allocated()

    def same(a, b):
        return a is not None and b is not None and torch.equal(a, b)

    checks = {
        "video_velocity_bitwise": same(reference["video"], observed["video"]),
        "audio_velocity_bitwise": same(reference["audio"], observed["audio"]),
    }
    if layout is None or layout.stage.first:  # the first stage alone embeds the inputs
        for k in INPUTS:
            checks[f"input_grad.{k}"] = same(
                reference["input_grads"][k], observed["input_grads"].get(k)
            )
        if args.train_time_embedder:
            observed_time = _time_embedder_grads(model)
            checks["time_embedder_grads"] = set(observed_time) == set(
                reference["time_embedder"]
            ) and all(same(v, observed_time[k]) for k, v in reference["time_embedder"].items())
            checks["time_embedder_grads"] &= len(observed_time) == 4
    if layout is not None:
        checks["handoffs_verified"] = observed["refused"] is None
    forward_order = [f"block.{i}" for i in range(50)]
    if layout is None:
        drift_forward = pm.first_drift(
            reference["snapshots"], observed["snapshots"], spans, tags, forward_order
        )
        drift_backward = pm.first_drift(
            reference["snapshots"],
            observed["snapshots"],
            spans,
            tags,
            [f"block.{i}.grad" for i in reversed(range(50))],
        )
    else:
        drift_forward, drift_backward = (
            pipe.stage_drift(
                reference["snapshots"], observed["snapshots"], layout, spans, tags, backward=back
            )
            for back in (False, True)
        )
    checks["block_outputs_bitwise"] = drift_forward is None
    checks["block_output_gradients_bitwise"] = drift_backward is None
    verdict = shard_gradient_verdict(
        expected,
        observed["leaf_digests"],
        len(model.transformer_blocks) if layout is None else layout.stage.hi - layout.stage.lo,
        lora=bool(args.lora_rank),
    )
    base_mismatched, lora_mismatched = verdict["base_mismatched"], verdict["lora_mismatched"]
    checks["weight_shard_gradients_bitwise"] = verdict["base_bitwise"]
    if args.lora_rank:
        checks["lora_gradients_bitwise"] = verdict["lora_bitwise"]
    result = {
        "rank": rank,
        "host": socket.gethostname(),
        "checks": checks,
        "weight_shards_compared": verdict["base_compared"],
        "weight_shards_mismatched": base_mismatched[:10],
        "lora_shards_compared": verdict["lora_compared"],
        "lora_shards_mismatched": lora_mismatched[:10],
        "shards_missing": verdict["missing"][:10],
        "lora_bytes": lora_bytes or None,
        "first_drift_forward": drift_forward,
        "first_drift_backward": drift_backward,
        "seconds": {
            "load": load_seconds,
            "canonical": reference["seconds"],
            "parallel": observed["seconds"],
        },
        "readback": (
            pm.grid_readback(grid, sp=args.sp, ops=ops) if layout is None else observed["readback"]
        ),
        "sharded": sharded,
        **reports,
    }
    if layout is not None:
        result["stage"] = vars(layout.stage)
    gathered = [None] * world
    dist.all_gather_object(gathered, result)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        merged = {}
        if layout is not None:
            pipe.check_coverage([r["stage"] for r in gathered], 50)
            merged = {
                "pipeline_first_drift_forward": pipe.merge_drift(
                    [r["first_drift_forward"] for r in gathered]
                ),
                "pipeline_first_drift_backward": pipe.merge_drift(
                    [r["first_drift_backward"] for r in gathered], backward=True
                ),
            }
        report = dict(
            contract=pm.CONTRACT if layout is None else pipe.CONTRACT,
            tp=args.tp,
            cp=args.cp,
            pp=args.pp,
            pp_sizes=[r["stage"]["hi"] - r["stage"]["lo"] for r in gathered] if layout else None,
            inject_handoff=args.inject_handoff or None,
            train_time_embedder=args.train_time_embedder,
            torch_deterministic_algorithms=deterministic,
            **merged,
            sp=args.sp,
            cp_mode=args.cp_mode,
            attention=args.attention,
            size=args.size,
            packed_rows=length,
            inject=args.inject or None,
            sharded=args.sharded,
            lora=(
                dict(contract=pl.CONTRACT, **lora_kw, scale=args.lora_alpha / args.lora_rank)
                if args.lora_rank
                else None
            ),
            strict_boundaries=strict,
            world=world,
            spans=spans,
            all_bitwise=all(all(r["checks"].values()) for r in gathered),
            ranks=gathered,
            torch=torch.__version__,
            device=torch.cuda.get_device_name(),
        )
        (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        print(
            json.dumps(
                {k: report[k] for k in ("pp", "tp", "cp", "sp", "cp_mode", "size", "all_bitwise")}
            )
        )
    dist.barrier()
    if rank == 0:
        time.sleep(3)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
