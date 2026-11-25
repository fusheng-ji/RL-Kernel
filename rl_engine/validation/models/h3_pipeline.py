# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pipeline parallelism for the H3 transformer (``h3-pipeline-v1``, RFC #420
``ws2_pipeline_parallel``): a layer partition with exact activation handoff.

The world is ``pp`` stages of ``tp * cp`` consecutive ranks. Each stage holds a contiguous range of
the 50 blocks and runs them on its own TP x CP (x SP) grid (``h3_tp.make_grid(pp=...)``); the first
stage also owns everything before block 0 and the last everything after block 49. Rank ``r`` of a
stage hands over to rank ``r`` of the next one: each such column is one pipeline of
``torch.distributed.pipelining`` (GPipe / 1F1B), which moves the tensors. This module adds what that
library does not have: the H3 stages, the composition with the grid, and the checks below.

What crosses a boundary, and why the result is the unsplit model's bit for bit:

* forward: the packed ``hidden_states`` (BF16) and ``temb`` (FP32), as they are (no cast, no
  arithmetic). Everything else a block reads (rotary tables, AdaLN row index) is computed on every
  stage from the request's positions, tags and timestep indices, which every rank holds.
* backward: the gradient of ``hidden_states`` and the *partial* gradient of ``temb``. In the unsplit
  model the ``temb`` gradient is one FP32 chain, ``norm_out + block 49 + ... + block 0``. A stage
  receives the partial sum of the later stages as the gradient of the ``temb`` it passed on, which
  autograd puts first, and its own blocks continue the chain in the same order. Summing each stage's
  contribution on its own and adding the sums afterwards is a different FP32 sum.
* several passes (micro-batches): their gradients accumulate in pass order on every stage, whatever
  the schedule interleaves.

Every handoff is checked: both ends reduce the tensor to ``h3_parallel_model.digest_tensor`` as it
leaves and as it is used, and after the passes the world agrees on them (a difference raises
``HandoffError`` on every rank). The ranks of a stage must also have received the same bytes, and
every rank must hold the same request.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from rl_engine.validation.models import h3_block as hb
from rl_engine.validation.models import h3_packed_forward as hpf
from rl_engine.validation.models import h3_parallel_model as pm
from rl_engine.validation.models import h3_tp

CONTRACT = "h3-pipeline-v1"
SCHEDULES = ("serial", "gpipe", "1f1b")
# Top-level modules with parameters that run before block 0 / after block 49.
FIRST_STAGE = ("proj_in", "audio_proj_in", "context_embedder", "token_refiner", "time_embedder")
LAST_STAGE = ("norm_out", "proj_out", "audio_proj_out")
# The request tensors of the transformer forward. Only the first three carry gradients.
REQUEST = (
    "hidden_states",
    "audio_hidden_states",
    "encoder_hidden_states",
    "timestep",
    "timestep_indices",
    "token_tags",
    "position_ids",
    "video_indices",
    "audio_indices",
    "text_indices",
)


class HandoffError(RuntimeError):
    """A boundary tensor did not arrive as it was sent, or ranks hold different requests."""


# --- partition -------------------------------------------------------------------------------


def block_sizes(total: int, pp: int, sizes: Sequence[int] | None = None) -> list[int]:
    """Blocks per stage: ``sizes`` if given, else even with the remainder on the first stages."""

    if pp < 1:
        raise ValueError(f"pp {pp} < 1")
    if sizes is None:
        if total < pp:
            raise ValueError(f"{total} blocks cannot fill {pp} stages")
        base, extra = divmod(total, pp)
        return [base + (s < extra) for s in range(pp)]
    out = [int(s) for s in sizes]
    if len(out) != pp or sum(out) != total or min(out) < 1:
        raise ValueError(f"stage sizes {out} are not {pp} positive sizes summing to {total}")
    return out


@dataclass(frozen=True)
class StageSpec:
    """Stage ``pp_rank`` of ``pp``: blocks ``lo <= i < hi`` of ``total``."""

    pp: int
    pp_rank: int
    lo: int
    hi: int
    total: int

    @classmethod
    def make(cls, total: int, pp: int, pp_rank: int, sizes: Sequence[int] | None = None):
        """The stage of an even split of ``total`` blocks, or of the split ``sizes``."""

        if not 0 <= pp_rank < pp:
            raise ValueError(f"no stage {pp_rank} of {pp}")
        split = block_sizes(total, pp, sizes)
        lo = sum(split[:pp_rank])
        return cls(pp, pp_rank, lo, lo + split[pp_rank], total)

    @property
    def first(self) -> bool:
        """Whether the stage runs everything before block 0."""

        return self.pp_rank == 0

    @property
    def last(self) -> bool:
        """Whether the stage runs everything after the last block."""

        return self.pp_rank == self.pp - 1

    def owns_block(self, index: int) -> bool:
        """Whether block ``index`` is on this stage."""

        return self.lo <= index < self.hi

    def owns_parameter(self, name: str) -> bool:
        """Whether parameter ``name`` (its name in the transformer) is on this stage."""

        head, _, rest = name.partition(".")
        if head == "transformer_blocks":
            return self.owns_block(int(rest.split(".", 1)[0]))
        if head in FIRST_STAGE:
            return self.first
        if head in LAST_STAGE:
            return self.last
        raise ValueError(f"{name}: no pipeline stage owns {head!r}")


# --- layout ----------------------------------------------------------------------------------


@dataclass
class Layout:
    """This rank's place in the pipeline: its stage, its stage's grid, its column's group."""

    stage: StageSpec
    grid: h3_tp.Grid
    column: Any = None  # the ranks of this (cp_rank, tp_rank) over the stages
    world: Any = None  # the group of every rank of the pipeline (``None``: the world)

    @property
    def stage_size(self) -> int:
        """Ranks per stage."""

        return self.grid.tp * self.grid.cp

    def column_ranks(self) -> list[int]:
        """Global ranks of this rank's column, in stage order."""

        return [s * self.stage_size + self.grid.rank for s in range(self.stage.pp)]


def make_layout(
    pp: int,
    tp: int = 1,
    cp: int = 1,
    *,
    total: int = 50,
    sizes: Sequence[int] | None = None,
    local_size: int | None = None,
    attention: str = "h3_full",
    cp_mode: str = "ring",
) -> Layout:
    """Global rank ``r`` is ``(pp_rank, cp_rank, tp_rank)`` with TP fastest; a stage is ``tp * cp``
    consecutive ranks.

    Every rank creates every group in one order: the grids' (``h3_tp.make_grid``), then one group
    per column. Bind ``device_id`` in ``init_process_group`` (as the H3 runners do), so that NCCL
    connects each of them here and not at a first, possibly one-sided, point-to-point call.
    """

    block_sizes(total, pp, sizes)  # refused on every rank before any group exists
    grid = h3_tp.make_grid(
        tp, cp, local_size=local_size, attention=attention, cp_mode=cp_mode, pp=pp
    )
    size = tp * cp
    columns = [dist.new_group([s * size + c for s in range(pp)]) for c in range(size)]
    pp_rank = dist.get_rank() // size
    stage = StageSpec.make(total, pp, pp_rank, sizes)
    return Layout(stage, grid, columns[grid.rank])


# --- one stage of the model ------------------------------------------------------------------


def stage_parts(model) -> hpf.PackedParts:
    """The provider forward in parts, with the scatter the execution context bound.

    A forward that is not a known provider forward, or that the caller wrapped on the instance, is
    refused: there would be no graph to split.
    """

    pack = model.__dict__.get(hpf.PACK_ATTR)
    if pack is None and "forward" in model.__dict__:
        raise hpf.UnknownForwardError("the model's forward is wrapped on the instance")
    return hpf.packed_parts(hpf.known_variant(type(model)), pack or hpf.provider_pack)


class _Handoff:
    """Digests of one direction's tensors at this rank, by ``(pass, boundary, name)``."""

    def __init__(self) -> None:
        self.sent: dict[tuple, torch.Tensor] = {}
        self.received: dict[tuple, torch.Tensor] = {}

    def to_lists(self) -> dict[str, dict[tuple, list[int]]]:
        """Both tables as Python integers (the only host synchronisation)."""

        return {
            side: {k: [int(x) for x in v] for k, v in getattr(self, side).items()}
            for side in ("sent", "received")
        }


@dataclass(frozen=True)
class Corruption:
    """A negative control: row ``row`` of ``name`` is perturbed as it enters ``boundary``'s reader.

    ``boundary`` is the block index the boundary precedes. ``wire`` perturbs before the reader's
    digest (a transfer error: the run must be refused); otherwise after it (an error inside the
    stage: the drift report must name it). ``grad`` perturbs the gradient on its way back instead.
    """

    boundary: int
    row: int
    wire: bool = False
    grad: bool = False

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        """``t`` with one BF16 step on that row (differentiable)."""

        scale = torch.ones_like(t)
        scale[:, self.row] = 1.0078125
        return t * scale


class _StageModule(torch.nn.Module):
    """One stage as the module ``PipelineStage`` drives: one call per pass, in pass order.

    The first stage's positional input is the pass number (it reads the request from ``passes``);
    the others' are ``(hidden_states, temb)``. It returns ``(hidden_states, temb)``, or the two
    velocities on the last stage.
    """

    def __init__(self, model, layout: Layout, passes, trace, corrupt: Corruption | None):
        super().__init__()
        # not a submodule: the pipeline library must not scale or reduce the model's gradients
        object.__setattr__(self, "model", model)
        self.stage, self.grid = layout.stage, layout.grid
        self.passes, self.trace, self.corrupt = passes, trace, corrupt
        self.parts = stage_parts(model)
        self.forward_digests, self.backward_digests = _Handoff(), _Handoff()
        self.calls = 0

    def _record(self, table: dict, key: tuple, t: torch.Tensor) -> None:
        table[key] = pm.digest_tensor(t)

    def _control(self, boundary: int, *, grad: bool) -> Corruption | None:
        """The negative control at ``boundary`` in this direction, if there is one."""

        c = self.corrupt
        return c if c is not None and c.boundary == boundary and c.grad == grad else None

    def _enter(self, n: int, hidden: torch.Tensor, temb: torch.Tensor):
        """Digest what arrived (perturbed in a negative control) and the gradients sent back."""

        boundary = self.stage.lo
        bad = self._control(boundary, grad=False)
        for name, leaf in (("hidden_states", hidden), ("temb", temb)):
            if leaf.requires_grad:

                def hook(p, name=name):
                    self._record(self.backward_digests.sent, (n, boundary, name), p.grad)

                leaf.register_post_accumulate_grad_hook(hook)
        received = self.forward_digests.received
        if bad is not None and bad.wire:
            hidden = bad.apply(hidden)
        self._record(received, (n, boundary, "hidden_states"), hidden)
        self._record(received, (n, boundary, "temb"), temb)
        if bad is not None and not bad.wire:
            hidden = bad.apply(hidden)
        self.trace.observe(f"handoff.{boundary}", hidden)
        return hidden, temb

    def _leave(self, n: int, hidden: torch.Tensor, temb: torch.Tensor):
        """Digest what leaves, and the gradients that come back (perturb them in a control).

        Both leave through an identity node of their own: it is the youngest node of the stage, so
        autograd runs it first and the received ``temb`` gradient heads the chain; and what the
        hooks change reaches every earlier hook.
        """

        boundary = self.stage.hi
        bad = self._control(boundary, grad=True)
        out = []
        for name, t in (("hidden_states", hidden), ("temb", temb)):
            self._record(self.forward_digests.sent, (n, boundary, name), t)
            t = t.view_as(t)
            if t.requires_grad:

                def hook(grad, name=name):
                    """Digest the gradient as it is used."""
                    key = (n, boundary, name)
                    received = self.backward_digests.received
                    if bad is None or name != "hidden_states":
                        self._record(received, key, grad)
                        return None
                    used = bad.apply(grad)
                    self._record(received, key, used if bad.wire else grad)
                    return used

                t.register_hook(hook)
            out.append(t)
        return out[0], out[1]

    def forward(self, *args):
        """One pass of this stage."""

        n = self.calls
        self.calls += 1
        inputs, model = self.passes[n], self.model
        if self.stage.first:
            hidden, temb, index, rope = self.parts.embed(model, inputs)
        else:
            hidden, temb = self._enter(n, *args)
            index, rope = self.parts.metadata(model, inputs)
        blocks = [model.transformer_blocks[i] for i in range(self.stage.lo, self.stage.hi)]
        hidden = hpf.run_blocks(model, blocks, hidden, temb, index, rope)
        if self.stage.last:
            return self.parts.head(model, hidden, temb, inputs)
        return self._leave(n, hidden, temb)


class SeededCotangent(torch.autograd.Function):
    """A scalar whose backward hands out given cotangents: ``autograd.backward`` as a loss."""

    @staticmethod
    def forward(ctx, video, audio, grad_video, grad_audio):
        """Keep the cotangents; the value itself means nothing (zero)."""

        ctx.save_for_backward(grad_video, grad_audio)
        return video.new_zeros((), dtype=torch.float32)

    @staticmethod
    def backward(ctx, _one):
        """The stored cotangents."""

        grad_video, grad_audio = ctx.saved_tensors
        return grad_video, grad_audio, None, None


# --- agreement -------------------------------------------------------------------------------


def _gather(layout: Layout, value: Any, group: Any = None) -> list[Any]:
    """``value`` of every rank of ``group`` (default: the pipeline), in rank order."""

    if not dist.is_initialized():
        return [value]
    everyone: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(everyone, value, group=group)
    return everyone


def check_requests(layout: Layout, passes: Sequence[dict[str, torch.Tensor]]) -> None:
    """Refuse, on every rank alike, passes that are not the same request everywhere."""

    mine: Any = None
    error = None
    try:
        for n, inputs in enumerate(passes):
            extra = set(inputs) - set(REQUEST) - {"return_dict"}
            missing = set(REQUEST) - set(inputs)
            if extra or missing:
                raise ValueError(f"pass {n}: unexpected {sorted(extra)}, missing {sorted(missing)}")
            hpf.check_inputs(inputs)
        mine = [
            {k: (str(p[k].dtype), tuple(p[k].shape), pm.exact_digest(p[k])) for k in REQUEST}
            for p in passes
        ]
    except (ValueError, TypeError) as exc:
        error = str(exc)
    everyone = _gather(layout, (error, mine), layout.world)
    errors = {r: e for r, (e, _) in enumerate(everyone) if e is not None}
    if errors:
        raise ValueError(f"requests refused: {errors}")
    for r, (_, theirs) in enumerate(everyone):
        if theirs != everyone[0][1]:
            raise HandoffError(f"rank {r} holds a different request than rank 0")


def handoff_problems(reports: Sequence[dict[str, Any]], pp: int) -> list[str]:
    """What is wrong with the handoffs of every rank's ``reports`` (empty: nothing).

    A report is ``{"stage", "position", "forward": _Handoff.to_lists(), "backward": ...}`` with
    ``position`` the rank's place in its stage. For every boundary: the two ends of each column
    name the same tensors and digests, and every column carries the same ones.
    """

    problems = []
    by_place = {(r["stage"], r["position"]): r for r in reports}
    positions = sorted({r["position"] for r in reports})
    for s in range(pp - 1):
        for direction, src, dst in (("forward", s, s + 1), ("backward", s + 1, s)):
            first: dict | None = None
            for c in positions:
                ends = by_place.get((src, c)), by_place.get((dst, c))
                if ends[0] is None or ends[1] is None:
                    problems.append(f"{direction} {src}->{dst}: no report for position {c}")
                    continue
                sent = _at(ends[0][direction]["sent"], ends[0], ends[1])
                received = _at(ends[1][direction]["received"], ends[0], ends[1])
                for key in sorted(set(sent) ^ set(received)):
                    problems.append(
                        f"{direction} {src}->{dst} position {c}: {key} has only one end"
                    )
                for key in sorted(set(sent) & set(received)):
                    if sent[key] != received[key]:
                        problems.append(
                            f"{direction} {src}->{dst} position {c}: pass {key[0]} {key[2]} at "
                            f"block {key[1]} arrived changed"
                        )
                if first is None:
                    first = sent
                elif sent != first:
                    problems.append(
                        f"{direction} {src}->{dst}: position {c} sent other bytes than "
                        f"position {positions[0]}"
                    )
                if direction == "forward" and not sent:
                    problems.append(f"{direction} {src}->{dst} position {c}: nothing was sent")
    return problems


def _at(table: dict[tuple, list[int]], a: dict, b: dict) -> dict[tuple, list[int]]:
    """The entries of ``table`` at the boundary between the stages of reports ``a`` and ``b``."""

    boundary = max(a["lo"], b["lo"])
    return {k: v for k, v in table.items() if k[1] == boundary}


# --- running ---------------------------------------------------------------------------------


def _meta(shape, dtype, requires_grad: bool) -> torch.Tensor:
    """An example tensor of a boundary (the library reads its shape, dtype and grad flag)."""

    return torch.empty(shape, dtype=dtype, device="meta").requires_grad_(requires_grad)


@dataclass(frozen=True)
class _Shape:
    """What fixes a schedule's buffers: passes of equal shape can share one."""

    rows: int
    timesteps: int
    video: tuple[int, ...]
    audio: tuple[int, ...]


class Pipeline:
    """The H3 transformer on this rank's stage (see the module docstring).

    ``model`` holds at least the stage's parameters (``h3_sharded_load.load_sharded_model(stage=)``
    for a model that holds only those). ``run`` takes the passes of one call.
    """

    def __init__(
        self,
        model,
        layout: Layout,
        *,
        sp: bool,
        strict_boundaries: bool = True,
        ops: hb.CandidateOps | None = None,
    ):
        total = len(model.transformer_blocks)
        if layout.stage.total != total:
            raise ValueError(
                f"the layout splits {layout.stage.total} blocks, the model has {total}"
            )
        self.model, self.layout, self.sp = model, layout, sp
        self.strict_boundaries = strict_boundaries
        self._ops = ops
        self.corrupt: Corruption | None = None  # a negative control only
        self.handoffs: list[dict[str, Any]] = []  # every rank's report of the last ``run``

    @property
    def ops(self) -> hb.CandidateOps:
        """The block operators, resolved on first use (the CUDA kernel rows)."""

        if self._ops is None:
            self._ops = hb.CandidateOps()
        return self._ops

    # --- what is refused ------------------------------------------------------------------

    def _temb_requires_grad(self, grad: bool) -> bool:
        """Whether ``temb`` carries a gradient: decided by stage 0, agreed by every rank."""

        mine = grad and any(p.requires_grad for p in self.model.time_embedder.parameters())
        flags = _gather(self.layout, (self.layout.stage.first, mine), self.layout.world)
        first = {f for is_first, f in flags if is_first}
        if len(first) != 1:
            raise ValueError("the first stage's ranks disagree on a trainable time_embedder")
        return first.pop()

    def _check(self, passes, loss_fn, schedule: str, temb_grad: bool) -> None:
        grid = self.layout.grid
        if schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {schedule!r}")
        if not passes:
            raise ValueError("no passes")
        if temb_grad and (grid.cp > 1 or self.sp):
            # under CP / SP a rank's blocks hold its rows only, so its ``temb`` gradient is partial
            # (``norm_out``'s is not): no chain over the stages would be the unsplit gradient
            raise ValueError("a trainable time_embedder needs cp=1 and no sequence parallelism")
        if len(passes) > 1 and loss_fn is not None and self._keyed_rows_train():
            raise ValueError(
                "one canonical backward session holds one pass of the keyed rows: freeze proj_in / "
                "proj_out, or run one pass per call"
            )

    def _keyed_rows_train(self) -> bool:
        if not self.strict_boundaries:
            return False
        mods = ("proj_in", "proj_out", "token_refiner")
        return any(p.requires_grad for m in mods for p in getattr(self.model, m).parameters())

    # --- the execution context --------------------------------------------------------------

    @contextmanager
    def execution(self, trace):
        """The grid bindings (``h3_parallel_model``) around every forward and backward of a call."""

        stage = self.layout.stage
        with pm.parallel_model_execution(
            self.model,
            self.layout.grid,
            sp=self.sp,
            trace=trace,
            ops=self.ops,
            strict_boundaries=self.strict_boundaries,
        ) as state:
            if not stage.first:  # this stage runs no pack
                trace.packed_bindings["packed_index_copy"] = "other stage"
            if not stage.last:
                trace.packed_bindings["packed_index_select"] = "other stage"
            for name, binding in trace.boundary_bindings.items():
                if not (stage.first if name in FIRST_STAGE else stage.last):
                    binding["status"] = "other stage"
            yield state

    # --- passes -----------------------------------------------------------------------------

    def _shape(self, inputs) -> _Shape:
        batch = inputs["hidden_states"].shape[0]
        return _Shape(
            int(inputs["position_ids"].shape[0]),
            int(inputs["timestep"].shape[0]),
            (batch, int(inputs["video_indices"].numel()), self.model.proj_out.out_features),
            (batch, int(inputs["audio_indices"].numel()), self.model.audio_proj_out.out_features),
        )

    def _groups(self, passes, schedule: str) -> list[tuple[int, int, str]]:
        """Consecutive passes that share one schedule: ``(first, count, "gpipe" | "1f1b")``."""

        out: list[tuple[int, int, str]] = []
        start = 0
        shapes = [self._shape(p) for p in passes]
        for i in range(1, len(passes) + 1):
            if i < len(passes) and schedule != "serial" and shapes[i] == shapes[start]:
                continue
            count = i - start
            # 1F1B needs a pass per stage in flight; a shorter group is GPipe (same bytes)
            kind = "1f1b" if schedule == "1f1b" and count >= self.layout.stage.pp else "gpipe"
            out.append((start, count, kind))
            start = i
        return out

    def run(
        self,
        passes: Sequence[dict[str, torch.Tensor]],
        loss_fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        *,
        trace,
        schedule: str = "serial",
        configure: Callable[[pm.ParallelState], None] | None = None,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor]]:
        """Run the passes; returns every pass's ``(video, audio)`` and loss, on every rank.

        With ``loss_fn(n, video, audio)`` (a scalar of pass ``n``, evaluated on the last stage) each
        pass also runs its backward, in pass order; without it nothing is differentiated.
        ``configure`` sets up the grid state (gradient sinks, a negative control) before the first
        pass. Raises ``HandoffError`` on every rank if a boundary tensor changed on its way.
        """

        passes = [{k: p[k] for k in p if k != "return_dict"} for p in passes]
        layout, stage = self.layout, self.layout.stage
        check_requests(layout, passes)
        temb_grad = self._temb_requires_grad(loss_fn is not None)
        self._check(passes, loss_fn, schedule, temb_grad)
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        losses: list[torch.Tensor] = []
        grad = torch.enable_grad() if loss_fn is not None else torch.no_grad()
        with grad, self.execution(trace) as state:
            if configure is not None:
                configure(state)
            module = _StageModule(self.model, layout, passes, trace, self.corrupt)
            for first, count, kind in self._groups(passes, schedule):
                out, loss = self._run_group(
                    module, passes[first], first, count, kind, loss_fn, temb_grad
                )
                outputs += out
                losses += loss
        self._verify(module)
        if stage.pp > 1:
            outputs, losses = self._share(passes, outputs, losses, loss_fn is not None)
        return outputs, losses

    def _run_group(self, module, example, first, count, kind, loss_fn, temb_grad):
        """One schedule over passes ``first .. first + count - 1`` (equal shapes)."""

        from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

        stage, device = self.layout.stage, example["timestep"].device
        shape, hidden = self._shape(example), self.model.config.hidden_size
        outs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        losses: list[torch.Tensor] = []
        numbers = torch.arange(first, first + count, device=device).view(count, 1)

        def loss_of(output, number):
            n = int(number)
            video, audio = output
            outs[n] = (video.detach(), audio.detach())
            loss = loss_fn(n, video, audio)
            if loss.dim() != 0:
                raise ValueError("loss_fn must return a scalar")
            return loss

        if stage.pp == 1:  # nothing to hand over: the same module, without the library
            for n in range(first, first + count):
                output = module(numbers[n - first])
                if loss_fn is None:
                    outs[n] = output
                else:
                    loss = loss_of(output, torch.tensor(n))
                    loss.backward()
                    losses.append(loss.detach())
            return [outs[n] for n in sorted(outs)], losses

        grad = loss_fn is not None
        # the text stream sets the packed sequence's dtype; the heads' weights the velocities'
        boundary = (
            _meta(
                (shape.video[0], shape.rows, hidden), self.model.context_embedder.weight.dtype, grad
            ),
            _meta((shape.timesteps, self.model.config.time_embed_dim), torch.float32, temb_grad),
        )
        velocities = (
            _meta(shape.video, self.model.proj_out.weight.dtype, grad),
            _meta(shape.audio, self.model.audio_proj_out.weight.dtype, grad),
        )
        pipe = PipelineStage(
            module,
            stage.pp_rank,
            stage.pp,
            device,
            input_args=(_meta((1, 1), torch.int64, False),) if stage.first else boundary,
            output_args=velocities if stage.last else boundary,
            group=self.layout.column,
        )
        cls = Schedule1F1B if kind == "1f1b" else ScheduleGPipe
        schedule = cls(
            pipe, n_microbatches=count, loss_fn=loss_of if grad else None, scale_grads=False
        )
        args = (numbers,) if stage.first else ()
        kwargs: dict[str, Any] = {}
        if grad and stage.last:
            kwargs.update(target=numbers, losses=losses)
        merged = schedule.step(*args, **kwargs, return_outputs=not grad)
        if stage.last and not grad:
            video, audio = merged
            for i in range(count):
                outs[first + i] = (
                    video[i * shape.video[0] : (i + 1) * shape.video[0]],
                    audio[i * shape.audio[0] : (i + 1) * shape.audio[0]],
                )
        return [outs[n] for n in sorted(outs)], [x.detach() for x in losses]

    def _verify(self, module: _StageModule) -> None:
        """Every rank's handoff digests, agreed by the world."""

        layout = self.layout
        report = {
            "stage": layout.stage.pp_rank,
            "position": layout.grid.rank,
            "lo": layout.stage.lo,
            "hi": layout.stage.hi,
            "forward": module.forward_digests.to_lists(),
            "backward": module.backward_digests.to_lists(),
        }
        if layout.stage.pp == 1:
            self.handoffs = [report]
            return
        self.handoffs = _gather(layout, report, layout.world)
        problems = handoff_problems(self.handoffs, layout.stage.pp)
        if problems:
            raise HandoffError("; ".join(problems[:8]))

    def _share(self, passes, outputs, losses, grad: bool):
        """The last stage's velocities and losses on every rank of the column (copies)."""

        layout = self.layout
        src, device = layout.column_ranks()[-1], passes[0]["timestep"].device
        shared = []
        for n, inputs in enumerate(passes):
            shape = self._shape(inputs)
            pair = []
            heads = (self.model.proj_out, self.model.audio_proj_out)
            for k, size in enumerate((shape.video, shape.audio)):
                t = (
                    outputs[n][k].detach().contiguous()
                    if layout.stage.last
                    else torch.empty(size, dtype=heads[k].weight.dtype, device=device)
                )
                dist.broadcast(t, src=src, group=layout.column)
                pair.append(t)
            shared.append((pair[0], pair[1]))
        if not grad:
            return shared, []
        values = torch.empty(len(passes), dtype=torch.float32, device=device)
        if layout.stage.last:
            values = torch.stack([x.to(torch.float32).to(device) for x in losses])
        dist.broadcast(values, src=src, group=layout.column)
        return shared, list(values.unbind())

    def readback(self) -> dict[str, Any]:
        """What this rank runs, for evidence."""

        layout, stage = self.layout, self.layout.stage
        return {
            "contract": CONTRACT,
            "transport": f"torch.distributed.pipelining {torch.__version__}",
            "pp": stage.pp,
            "pp_rank": stage.pp_rank,
            "blocks": [stage.lo, stage.hi],
            "column_ranks": layout.column_ranks(),
            "grid": pm.grid_readback(layout.grid, sp=self.sp, ops=self.ops),
        }


# --- evidence --------------------------------------------------------------------------------


def stage_nodes(stage: StageSpec, *, backward: bool = False) -> list[str]:
    """The trace nodes a stage must leave, in execution order."""

    nodes = [] if stage.first else [f"handoff.{stage.lo}"]
    nodes += [f"block.{i}" for i in range(stage.lo, stage.hi)]
    return [n + ".grad" for n in reversed(nodes)] if backward else nodes


def stage_drift(
    reference: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    layout: Layout,
    spans: list[tuple[int, int]],
    token_tags: torch.Tensor | None = None,
    *,
    backward: bool = False,
) -> dict[str, Any] | None:
    """``h3_parallel_model.first_drift`` over this stage's nodes, with global owner ranks.

    ``reference`` are the unsplit run's snapshots: what enters boundary ``k`` is block ``k - 1``'s
    output (and its gradient on the way back). A handoff is held whole by each rank of the stage,
    so this rank is its owner.
    """

    stage, grid = layout.stage, layout.grid
    nodes = stage_nodes(stage, backward=backward)
    missing = [n for n in nodes if n not in observed]
    if missing:
        raise ValueError(f"the stage left no snapshot of {missing}")
    expected = {}
    for n in nodes:
        source = n
        if n.startswith("handoff."):
            source = f"block.{stage.lo - 1}" + (".grad" if n.endswith(".grad") else "")
        expected[n] = reference[source]
    found = pm.first_drift(expected, {n: observed[n] for n in nodes}, spans, token_tags, nodes)
    if found is None:
        return None
    if found["node"].startswith("handoff."):
        found["owner_ranks"] = [grid.first_rank + grid.rank]
    elif "owner_ranks" in found:
        found["owner_ranks"] = [grid.first_rank + r for r in found["owner_ranks"]]
    found["order"] = nodes.index(found["node"])
    found["stage"] = stage.pp_rank
    return found


def merge_drift(
    reports: Sequence[dict[str, Any] | None], *, backward: bool = False
) -> dict[str, Any] | None:
    """The first drift of the whole model from every rank's ``stage_drift``.

    Forward, the earliest stage and node; backward, the latest stage first. The owners of the
    same node and row on several ranks are merged.
    """

    found = [r for r in reports if r is not None]
    if not found:
        return None
    stage = (max if backward else min)(r["stage"] for r in found)
    in_stage = [r for r in found if r["stage"] == stage]
    order = min(r["order"] for r in in_stage)
    at = [r for r in in_stage if r["order"] == order]
    row = min((r["first_row"] for r in at if "first_row" in r), default=None)
    best = next((r for r in at if r.get("first_row") == row), at[0])
    merged = dict(best)
    owners = {o for r in at if r.get("first_row") == row for o in r.get("owner_ranks", [])}
    if owners:
        merged["owner_ranks"] = sorted(owners)
    return merged


def check_coverage(stages: Sequence[dict[str, int]], total: int) -> None:
    """Refuse a set of per-rank stages that does not hold every block on exactly one stage."""

    ranges = {(s["pp_rank"], s["lo"], s["hi"]) for s in stages}
    blocks = sorted(i for _, lo, hi in ranges for i in range(lo, hi))
    if blocks != list(range(total)):
        raise ValueError(f"the stages hold blocks {blocks}, not 0..{total - 1} once each")


def drop_off_stage_lora(model, stage: StageSpec) -> None:
    """Remove the block LoRA of the other stages (``h3_lora.attach_lora`` fills every block)."""

    for index, block in enumerate(model.transformer_blocks):
        if not stage.owns_block(index):
            block.__dict__.pop(pm.LORA_ATTR, None)
