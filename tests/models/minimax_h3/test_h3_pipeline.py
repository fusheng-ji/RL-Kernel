# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3-pipeline-v1``: the stages, the handoff and its checks.

The CPU tests run the Diffusers H3 transformer (small) as real gloo stages on
``torch.distributed.pipelining`` and compare outputs and every gradient with the unsplit model bit
for bit. The stages on the H3 kernels and the real 50 blocks are exercised by
``tools/validation/models/h3_parallel_full_model.py --pp``; the block chain below runs on GPUs.
"""

from __future__ import annotations

import os
import queue as queue_mod
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_engine.validation.models import h3_parallel_model as pm
from rl_engine.validation.models import h3_pipeline as pipe
from rl_engine.validation.models import h3_tp

transformer = pytest.importorskip(
    "diffusers.models.transformers.transformer_minimax_h3",
    reason="the Diffusers MiniMax-H3 transformer is the model",
)
pytest.importorskip("torch.distributed.pipelining")

LAYERS = 5
TINY = dict(
    num_attention_heads=2,
    attention_head_dim=16,
    hidden_size=32,
    num_layers=LAYERS,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=4,
    audio_in_channels=3,
    patch_size=(1, 2, 2),
    text_dim=8,
    freq_dim=16,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=2,
)


def tiny_model(dtype=torch.bfloat16):
    """The Diffusers transformer, small, seeded, FP32 boundaries and ``dtype`` blocks."""
    torch.manual_seed(0)
    model = transformer.MiniMaxH3Transformer3DModel(**TINY).to(dtype)
    for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"):
        getattr(model, name).to(torch.float32)
    model.rope.inv_freq = model.rope.inv_freq.float()
    return model.eval()


def request(seed=0, text=3, video=6, audio=2):
    """One packed request: text, video and audio rows, two timesteps."""
    gen = torch.Generator().manual_seed(seed)
    seq = text + video + audio
    tags = torch.tensor([1] * text + [0] * video + [2] * audio)
    return dict(
        hidden_states=torch.randn(1, video, 16, generator=gen),
        audio_hidden_states=torch.randn(1, audio, 3, generator=gen),
        encoder_hidden_states=torch.randn(1, text, 8, generator=gen),
        timestep=torch.tensor([0.25, 0.5 + seed / 100]),
        timestep_indices=(tags != 1).long(),
        token_tags=tags,
        position_ids=torch.arange(seq * 3).view(seq, 3),
        video_indices=torch.arange(text, text + video),
        audio_indices=torch.arange(text + video, seq),
        text_indices=torch.arange(text),
    )


LEAVES = ("hidden_states", "audio_hidden_states", "encoder_hidden_states")


def loss_fn(n, video, audio):
    """A scalar with a seeded FP32 cotangent per pass."""
    gen = torch.Generator().manual_seed(1000 + n)
    gv, ga = torch.randn(video.shape, generator=gen), torch.randn(audio.shape, generator=gen)
    return (video.float() * gv).sum() + (audio.float() * ga).sum()


def with_grads(passes):
    return [
        {k: (v.clone().requires_grad_() if k in LEAVES else v) for k, v in p.items()}
        for p in passes
    ]


def unsplit(model, passes):
    """The unsplit provider forward and backward of every pass, in order."""
    passes = with_grads(passes)
    outs, losses = [], []
    for n, p in enumerate(passes):
        video, audio = model(**p, return_dict=False)
        loss = loss_fn(n, video, audio)
        loss.backward()
        outs.append((video.detach(), audio.detach()))
        losses.append(loss.detach())
    return dict(
        outs=outs,
        losses=losses,
        params={k: v.grad for k, v in model.named_parameters()},
        inputs=[{k: p[k].grad for k in LEAVES} for p in passes],
    )


class ProviderPipeline(pipe.Pipeline):
    """The pipeline with the provider's own blocks: no CUDA kernels, the same handoffs."""

    @contextmanager
    def execution(self, trace):
        yield SimpleNamespace()


class Trace:
    """The ``observe`` of ``FullModelTrace``: forward and gradient snapshots."""

    def __init__(self):
        self.snapshots = {}

    def observe(self, name, value):
        self.snapshots[name] = value.detach().clone()
        if value.requires_grad:
            value.register_hook(
                lambda g, name=name: self.snapshots.__setitem__(name + ".grad", g.clone())
            )


def layout_of(pp, sizes=None):
    """One rank per stage, on gloo (no TP / CP, so no CUDA collective is built)."""
    import torch.distributed as dist

    rank = dist.get_rank()
    column = dist.new_group(list(range(pp)))
    stage = pipe.StageSpec.make(LAYERS, pp, rank, sizes)
    grid = h3_tp.Grid(1, 1, 0, 0, group=None, first_rank=rank)
    return pipe.Layout(stage, grid, column)


def unload_off_stage(model, stage):
    """What the sharded loader leaves of the other stages: trainable ``meta`` placeholders.

    FP64 (the loader leaves the default dtype, which may happen to be the right one), so any use of
    a placeholder's dtype shows.
    """
    for name, p in list(model.named_parameters()):
        if not stage.owns_parameter(name):
            owner, _, leaf = name.rpartition(".")
            empty = torch.empty(p.shape, dtype=torch.float64, device="meta")
            setattr(model.get_submodule(owner), leaf, torch.nn.Parameter(empty))


# --- real gloo ranks -----------------------------------------------------------------------------


def _entry(rank, world, init, out, target, kwargs):
    import torch.distributed as dist

    torch.set_num_threads(1)
    try:
        dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
        out.put((rank, target(rank, world, **kwargs)))
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        out.put((rank, RuntimeError(traceback.format_exc())))
        os._exit(1)


def run_ranks(world, target, **kwargs):
    """``target(rank, world, **kwargs)`` on ``world`` gloo processes; results by rank."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("forkserver")  # one process imports the modules, the ranks fork from it
    ctx.set_forkserver_preload(
        [
            "rl_engine.validation.models.h3_pipeline",
            "diffusers.models.transformers.transformer_minimax_h3",
            "torch.distributed.pipelining",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        init, out = (Path(tmp) / "init").as_uri(), ctx.Queue()
        procs = [
            ctx.Process(target=_entry, args=(r, world, init, out, target, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        results = {}
        try:
            while len(results) < world:
                try:
                    rank, value = out.get(timeout=180)
                except queue_mod.Empty:
                    raise RuntimeError(f"only ranks {sorted(results)} reported") from None
                if isinstance(value, Exception):
                    raise value
                results[rank] = value
        finally:
            for p in procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()
        return [results[r] for r in range(world)]


def _staged(rank, world, *, sizes, checkpoint, schedule, lengths, dtype, frozen_time):
    """Every pass on this rank's stage against the unsplit model; the names that differ."""
    passes = [request(n, text=t) for n, t in enumerate(lengths)]
    ref_model, model = tiny_model(dtype), tiny_model(dtype)
    for m in (ref_model, model):
        for p in m.time_embedder.parameters():
            p.requires_grad_(not frozen_time)
        if checkpoint:
            m.enable_gradient_checkpointing()
    reference = unsplit(ref_model, passes)
    layout = layout_of(world, sizes)
    unload_off_stage(model, layout.stage)
    runner = ProviderPipeline(model, layout, sp=False, strict_boundaries=False)
    mine = with_grads(passes)
    outs, losses = runner.run(mine, loss_fn, trace=Trace(), schedule=schedule)
    bad = []
    for n, (got, want) in enumerate(zip(outs, reference["outs"])):
        bad += [f"out{n}.{k}" for k in (0, 1) if not torch.equal(got[k], want[k])]
    bad += [
        f"loss{n}" for n, (a, b) in enumerate(zip(losses, reference["losses"])) if float(a) != b
    ]
    for name, p in model.named_parameters():
        want = reference["params"][name]
        if not layout.stage.owns_parameter(name):
            if p.grad is not None:
                bad.append(f"{name} has a gradient on a stage that does not own it")
        elif (p.grad is None) != (want is None) or (
            want is not None and not torch.equal(p.grad, want)
        ):
            bad.append(name)
    if layout.stage.first:
        for n, p in enumerate(mine):
            bad += [
                f"in{n}.{k}"
                for k in LEAVES
                if not torch.equal(p[k].grad, reference["inputs"][n][k])
            ]
    with torch.no_grad():  # the forward-only replay
        again, none = runner.run(passes, trace=Trace(), schedule=schedule)
    for n, (got, want) in enumerate(zip(again, reference["outs"])):
        bad += [f"replay{n}.{k}" for k in (0, 1) if not torch.equal(got[k], want[k])]
    if none:
        bad.append("a forward-only run returned losses")
    return bad


class TestStages:
    @pytest.mark.parametrize(
        ("pp", "sizes", "expected"),
        [
            (1, None, [50]),
            (2, None, [25, 25]),
            (4, None, [13, 13, 12, 12]),
            (8, None, [7, 7, 6, 6, 6, 6, 6, 6]),
            (3, [1, 48, 1], [1, 48, 1]),
        ],
    )
    def test_the_split(self, pp, sizes, expected):
        """Contiguous ranges that hold every block once; the first stages take the remainder."""
        assert pipe.block_sizes(50, pp, sizes) == expected
        stages = [pipe.StageSpec.make(50, pp, r, sizes) for r in range(pp)]
        assert [s.hi - s.lo for s in stages] == expected
        assert [i for s in stages for i in range(s.lo, s.hi)] == list(range(50))
        assert [s.first for s in stages] == [True] + [False] * (pp - 1)
        assert [s.last for s in stages] == [False] * (pp - 1) + [True]
        pipe.check_coverage([vars(s) for s in stages], 50)

    @pytest.mark.parametrize(
        ("pp", "sizes"),
        [(0, None), (51, None), (2, [25, 24]), (2, [50]), (2, [50, 0]), (2, [51, -1])],
    )
    def test_a_split_that_is_not_one_is_refused(self, pp, sizes):
        """Wrong sums, counts or empty stages raise."""
        with pytest.raises(ValueError):
            pipe.block_sizes(50, pp, sizes)

    def test_every_parameter_of_the_model_is_on_exactly_one_stage(self):
        """The stages partition the transformer's parameters; an unknown module is refused."""
        stages = [pipe.StageSpec.make(LAYERS, 3, r) for r in range(3)]
        for name, _ in tiny_model().named_parameters():
            assert sum(s.owns_parameter(name) for s in stages) == 1, name
        with pytest.raises(ValueError, match="no pipeline stage owns 'extra'"):
            stages[0].owns_parameter("extra.weight")

    @pytest.mark.parametrize(
        "stages", [[(0, 0, 3)], [(0, 0, 3), (1, 2, 5)], [(0, 0, 2), (1, 3, 5)]]
    )
    def test_coverage_refuses_a_missing_or_repeated_block(self, stages):
        """A gap, an overlap or a missing stage is refused."""
        with pytest.raises(ValueError, match="once each"):
            pipe.check_coverage([dict(pp_rank=p, lo=a, hi=b) for p, a, b in stages], 5)


@pytest.mark.parametrize(
    ("pp", "sizes", "checkpoint", "schedule", "lengths", "dtype", "frozen_time"),
    [
        (2, None, True, "serial", [3], torch.bfloat16, False),
        (2, [1, 4], False, "1f1b", [3, 3, 3], torch.float32, False),
        (2, [4, 1], True, "gpipe", [3, 3, 5, 5], torch.bfloat16, True),
        (4, None, True, "1f1b", [3, 3, 3, 3, 3, 4], torch.bfloat16, False),
        (4, None, True, "serial", [3, 5], torch.float32, False),
        (4, None, False, "gpipe", [3, 3, 3], torch.bfloat16, True),
        (5, None, True, "1f1b", [3] * 6, torch.float32, False),
    ],
)
def test_the_staged_model_equals_the_unsplit_one_bitwise(
    pp, sizes, checkpoint, schedule, lengths, dtype, frozen_time
):
    """Outputs, losses, input gradients and every parameter gradient accumulated over the passes
    (the time embedder's is the chain through every stage), for each schedule."""
    results = run_ranks(
        pp,
        _staged,
        sizes=sizes,
        checkpoint=checkpoint,
        schedule=schedule,
        lengths=lengths,
        dtype=dtype,
        frozen_time=frozen_time,
    )
    assert results == [[]] * pp


def _keyed(rank, world, *, trainable):
    """The refusal of two strict passes on this rank (``None``: not refused), and both dtypes."""
    model = tiny_model().requires_grad_(False)
    model.proj_out.requires_grad_(trainable)  # on the last stage only
    layout = layout_of(world)
    unload_off_stage(model, layout.stage)
    runner = ProviderPipeline(model, layout, sp=False, strict_boundaries=True)
    facts = runner._facts(True)
    try:
        runner.run(with_grads([request(), request(1)]), loss_fn, trace=Trace())
        refused = None
    except ValueError as exc:
        refused = str(exc)
    return facts.hidden, facts.velocities, facts.temb_grad, refused


@pytest.mark.parametrize("trainable", [True, False])
def test_every_rank_learns_of_parameters_from_the_stage_that_owns_them(trainable):
    """A placeholder's dtype and ``requires_grad`` decide nothing, and a refusal is every rank's."""
    results = run_ranks(3, _keyed, trainable=trainable)
    for hidden, velocities, temb_grad, refused in results:
        assert hidden == torch.bfloat16 and velocities == (torch.float32, torch.float32)
        assert temb_grad is False
        if trainable:
            assert "one pass of the keyed rows" in refused
        else:
            assert refused is None


def test_adding_each_stages_temb_sum_afterwards_is_not_the_unsplit_gradient():
    """The negative control of the ``temb`` chain: per-stage sums added later differ in FP32."""
    gen = torch.Generator().manual_seed(3)
    temb = torch.randn(64, generator=gen)
    weights = torch.randn(8, 64, generator=gen)
    h0 = torch.randn(5, 64, generator=gen)

    def blocks(h, t, lo, hi):
        for w in weights[lo:hi]:
            h = torch.tanh(h + w) + 0.3 * h * torch.sin(t * w)
        return h

    leaf = temb.clone().requires_grad_()
    blocks(h0, leaf, 0, 8).pow(2).sum().backward()

    def staged(chain):
        bounds, stages, h = [(0, 2), (2, 4), (4, 6), (6, 8)], [], h0
        for lo, hi in bounds:
            hin, tin = h.detach().requires_grad_(), temb.clone().requires_grad_()
            h = blocks(hin, tin, lo, hi)
            stages.append((hin, tin, h))
        gh, gt = None, None
        for hin, tin, out in reversed(stages):
            if gh is None:
                out.pow(2).sum().backward()
            elif chain:
                torch.autograd.backward([out, tin.view_as(tin)], [gh, gt])
            else:
                out.backward(gh)
            gh, gt = hin.grad, (tin.grad if chain or gt is None else gt + tin.grad)
        return gt

    assert torch.equal(staged(True), leaf.grad)
    assert not torch.equal(staged(False), leaf.grad)


# --- the checks ------------------------------------------------------------------------------


def _control(rank, world, *, corrupt, differ=False, bad_schedule=False):
    """One pass with a negative control; what this rank raised and its drift reports."""
    model = tiny_model()
    layout = layout_of(world)
    p = request(0)
    if differ and rank == 1:
        p["hidden_states"] = p["hidden_states"] + 1e-3
    reference = tiny_model()
    trace_ref = Trace()
    handles = [
        b.register_forward_hook(lambda m, a, o, i=i: trace_ref.observe(f"block.{i}", o))
        for i, b in enumerate(reference.transformer_blocks)
    ]
    unsplit(reference, [request(0)])
    trace = Trace()
    handles += [
        b.register_forward_hook(lambda m, a, o, i=i: trace.observe(f"block.{i}", o))
        for i, b in enumerate(model.transformer_blocks)
    ]
    runner = ProviderPipeline(model, layout, sp=False, strict_boundaries=False)
    runner.corrupt = None if corrupt is None else pipe.Corruption(**corrupt)
    error = None
    try:
        runner.run([p], loss_fn, trace=trace, schedule="zigzag" if bad_schedule else "serial")
    except (pipe.HandoffError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    if differ or bad_schedule:
        return error, None, None
    rows = p["position_ids"].shape[0]
    drift = [
        pipe.stage_drift(
            trace_ref.snapshots, trace.snapshots, layout, [(0, rows)], p["token_tags"], backward=b
        )
        for b in (False, True)
    ]
    return error, drift[0], drift[1]


class TestChecks:
    def test_no_control_no_report(self):
        """The same run without a control raises nothing and drifts nowhere."""
        assert run_ranks(2, _control, corrupt=None) == [(None, None, None)] * 2

    def test_a_changed_forward_tensor_is_refused_on_every_rank(self):
        """A row changed on the wire: both ends raise and name the pass, tensor and boundary."""
        for error, _, _ in run_ranks(2, _control, corrupt=dict(boundary=3, row=4, wire=True)):
            assert error.startswith("HandoffError")
            assert "forward 0->1 position 0: pass 0 hidden_states at block 3 arrived" in error

    def test_a_changed_gradient_is_refused_on_every_rank(self):
        """The same for the gradient going back."""
        results = run_ranks(2, _control, corrupt=dict(boundary=3, row=4, wire=True, grad=True))
        for error, _, _ in results:
            assert error.startswith("HandoffError")
            assert "backward 1->0 position 0: pass 0 hidden_states at block 3 arrived" in error

    def test_a_row_changed_inside_the_stage_is_the_first_drift(self):
        """After the digest: nothing is refused, and the drift report names node, row, owner."""
        results = run_ranks(2, _control, corrupt=dict(boundary=3, row=4))
        assert [e for e, _, _ in results] == [None, None]
        merged = pipe.merge_drift([f for _, f, _ in results])
        assert (merged["node"], merged["first_row"], merged["owner_ranks"]) == ("handoff.3", 4, [1])
        assert merged["modality"] == "video" and merged["rows_differing"] == 1
        assert results[0][1] is None  # the first stage ran before it

    def test_a_gradient_row_changed_inside_the_stage_is_the_first_backward_drift(self):
        """The first backward drift is the gradient the earlier stage started from."""
        results = run_ranks(2, _control, corrupt=dict(boundary=3, row=4, grad=True))
        assert [e for e, _, _ in results] == [None, None]
        assert [f for _, f, _ in results] == [None, None]  # the forward is untouched
        merged = pipe.merge_drift([b for _, _, b in results], backward=True)
        assert (merged["node"], merged["first_row"], merged["owner_ranks"]) == (
            "block.2.grad",
            4,
            [0],
        )

    def test_ranks_with_different_requests_are_refused_before_anything_runs(self):
        """A request that differs on one rank raises on both."""
        for error, _, _ in run_ranks(2, _control, corrupt=None, differ=True):
            assert "HandoffError: rank 1 holds a different request" in error

    def test_an_unknown_schedule_is_refused(self):
        """A schedule name that is not one raises on every rank."""
        for error, _, _ in run_ranks(2, _control, corrupt=None, bad_schedule=True):
            assert "unknown schedule 'zigzag'" in error


def _report(stage, position, sent=None, received=None, back_sent=None, back_received=None):
    lo = 25 * stage
    return dict(
        stage=stage,
        position=position,
        lo=lo,
        hi=lo + 25,
        forward=dict(sent=sent or {}, received=received or {}),
        backward=dict(sent=back_sent or {}, received=back_received or {}),
    )


class TestHandoffProblems:
    KEY = (0, 25, "hidden_states")

    def _pair(self, positions=2):
        ok = {self.KEY: [1, 2]}
        return [_report(0, c, sent=ok, back_received=ok) for c in range(positions)] + [
            _report(1, c, received=ok, back_sent=ok) for c in range(positions)
        ]

    def test_matching_ends_are_fine(self):
        """Equal digests at both ends and across the columns: nothing to report."""
        assert pipe.handoff_problems(self._pair(), 2) == []

    def test_a_changed_digest_a_missing_end_and_a_lost_report(self):
        """Each defect is named."""
        reports = self._pair()
        reports[3]["forward"]["received"] = {self.KEY: [1, 3]}
        assert pipe.handoff_problems(reports, 2) == [
            "forward 0->1 position 1: pass 0 hidden_states at block 25 arrived changed"
        ]
        reports = self._pair()
        reports[3]["backward"]["sent"] = {}
        assert any("has only one end" in p for p in pipe.handoff_problems(reports, 2))
        assert any("no report for position 1" in p for p in pipe.handoff_problems(reports[:3], 2))

    def test_columns_that_carry_different_bytes(self):
        """Two columns that each arrived unchanged but differ from each other: replica drift."""
        reports = self._pair()
        for r in (reports[1], reports[3]):
            side = "sent" if r["stage"] == 0 else "received"
            r["forward"][side] = {self.KEY: [7, 7]}
        assert pipe.handoff_problems(reports, 2) == [
            "forward 0->1: position 1 sent other bytes than position 0"
        ]

    def test_nothing_sent_is_a_problem(self):
        """A boundary without any forward tensor never counts as verified."""
        reports = [_report(0, 0), _report(1, 0)]
        assert pipe.handoff_problems(reports, 2) == ["forward 0->1 position 0: nothing was sent"]


class TestMergeDrift:
    def test_forward_takes_the_earliest_stage_and_backward_the_latest(self):
        """The first drift is in execution order, not in rank order."""
        reports = [
            dict(node="block.30", stage=1, order=5, first_row=2, owner_ranks=[2]),
            None,
            dict(node="block.3", stage=0, order=3, first_row=9, owner_ranks=[0]),
        ]
        assert pipe.merge_drift(reports)["node"] == "block.3"
        assert pipe.merge_drift(reports, backward=True)["node"] == "block.30"
        assert pipe.merge_drift([None, None]) is None

    def test_owners_of_the_same_row_are_merged_and_the_first_row_wins(self):
        """Several ranks naming the same node: the first row, and all its owners."""
        reports = [
            dict(node="handoff.25", stage=1, order=0, first_row=7, owner_ranks=[3]),
            dict(node="handoff.25", stage=1, order=0, first_row=4, owner_ranks=[2]),
            dict(node="handoff.25", stage=1, order=0, first_row=4, owner_ranks=[1]),
        ]
        merged = pipe.merge_drift(reports)
        assert merged["first_row"] == 4 and merged["owner_ranks"] == [1, 2]

    def test_a_stage_without_a_snapshot_is_refused(self):
        """A node the stage should have left but did not is an error, not "no drift"."""
        layout = pipe.Layout(pipe.StageSpec(2, 1, 3, 5, 5), h3_tp.Grid(1, 1, 0, 0, first_rank=1))
        with pytest.raises(ValueError, match="no snapshot of .'handoff.3'"):
            pipe.stage_drift({}, {"block.3": torch.zeros(1)}, layout, [(0, 1)])


class TestGroups:
    @pytest.mark.parametrize("rank", range(8))
    def test_every_rank_creates_the_same_groups_in_the_same_order(self, monkeypatch, rank):
        """PP2 x CP2 x TP2: TP groups, CP groups, stages, columns; and this rank's own ones."""
        import torch.distributed as dist

        from rl_engine.distributed.algorithms import collectives

        made = []
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)
        monkeypatch.setattr(dist, "get_rank", lambda group=None: rank)
        monkeypatch.setattr(dist, "new_group", lambda ranks: made.append(list(ranks)) or ranks)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            collectives, "DeterministicCollective", lambda group, device, **kw: group
        )
        layout = pipe.make_layout(2, 2, 2, total=50)
        assert made == [
            *([0, 1], [2, 3], [4, 5], [6, 7]),  # TP
            *([0, 2], [1, 3], [4, 6], [5, 7]),  # CP
            *([0, 1, 2, 3], [4, 5, 6, 7]),  # the stages
            *([0, 4], [1, 5], [2, 6], [3, 7]),  # the columns
        ]
        grid = layout.grid
        assert rank in grid.tp_collective and rank in grid.cp_group and rank in grid.group
        assert rank in layout.column and layout.column_ranks() == layout.column
        assert grid.first_rank + grid.rank == rank
        assert layout.stage.pp_rank == rank // 4

    def test_a_world_that_is_not_the_product_is_refused(self, monkeypatch):
        """PP2 x TP2 x CP2 needs 8 ranks."""
        import torch.distributed as dist

        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)
        monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
        with pytest.raises(ValueError, match="world 4 != tp 2 x cp 2 x pp 2"):
            pipe.make_layout(2, 2, 2)


class TestRefusals:
    def _runner(self, *, sp=False, cp=1, strict=False, total=LAYERS):
        layout = pipe.Layout(
            pipe.StageSpec(1, 0, 0, total, total), h3_tp.Grid(1, cp, 0, 0, attention="h3_full")
        )
        return ProviderPipeline(tiny_model(), layout, sp=sp, strict_boundaries=strict)

    @pytest.mark.parametrize("kw", [dict(sp=True), dict(cp=2)])
    def test_a_trainable_time_embedder_needs_whole_rows(self, kw):
        """Under CP or SP the ``temb`` gradient is partial: refused before the forward."""
        with pytest.raises(ValueError, match="trainable time_embedder"):
            self._runner(**kw).run([request()], loss_fn, trace=Trace())

    def test_several_passes_need_frozen_keyed_rows(self):
        """One canonical session holds one pass of a trainable keyed row."""
        with pytest.raises(ValueError, match="one pass of the keyed rows"):
            self._runner(strict=True).run([request(), request(1)], loss_fn, trace=Trace())

    def test_a_layout_of_another_model_is_refused(self):
        """The layout must split this model's blocks."""
        with pytest.raises(ValueError, match="the layout splits 50 blocks, the model has 5"):
            self._runner(total=50)

    def test_a_wrapped_forward_is_refused(self):
        """There is no graph to split if the caller replaced the instance's forward."""
        runner = self._runner()
        runner.model.forward = lambda **kw: None
        with pytest.raises(Exception, match="wrapped on the instance"):
            runner.run([request()], loss_fn, trace=Trace())

    @pytest.mark.parametrize("drop", ["timestep", "token_tags"])
    def test_an_incomplete_request_is_refused(self, drop):
        """A missing request tensor is named."""
        bad = request()
        bad.pop(drop)
        with pytest.raises(ValueError, match=drop):
            self._runner().run([bad], loss_fn, trace=Trace())

    def test_one_stage_is_the_unsplit_model(self):
        """``pp=1`` runs without a process group and gives the unsplit bytes."""
        runner, passes = self._runner(), [request(), request(1)]
        reference = unsplit(tiny_model(), passes)
        outs, losses = runner.run(with_grads(passes), loss_fn, trace=Trace())
        for got, want in zip(outs, reference["outs"]):
            assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
        for name, p in runner.model.named_parameters():
            assert torch.equal(p.grad, reference["params"][name]), name


class TestHandoffArgument:
    def _parse(self, text):
        from tools.validation.models import h3_parallel_full_model as tool

        return tool.parse_handoff(text, [0, 13, 26, 38], 40)

    def test_the_control_names_a_stage_start_a_row_and_its_kind(self):
        """``BLOCK:ROW``, with ``grad`` and ``wire`` in any order."""
        assert self._parse("26:5") == pipe.Corruption(26, 5)
        assert self._parse("13:0:wire") == pipe.Corruption(13, 0, wire=True)
        assert self._parse("38:39:wire:grad") == pipe.Corruption(38, 39, wire=True, grad=True)

    @pytest.mark.parametrize(
        "text", ["26", "0:5", "25:5", "26:40", "26:-1", "26:5:both", "26:5:wire:wire"]
    )
    def test_what_would_perturb_nothing_is_refused(self, text):
        """A block no later stage starts at, a row outside the sequence, an unknown kind."""
        with pytest.raises(ValueError):
            self._parse(text)


def test_digest_tensor_is_exact_digest():
    """The tensor form gives the same two checksums."""
    x = torch.randn(33, 7, generator=torch.Generator().manual_seed(2)).bfloat16()
    assert pm.digest_tensor(x, chunk=16).tolist() == pm.exact_digest(x)


# --- the stages on the H3 grid kernels -----------------------------------------------------------

BLOCKS = 4


def gpus(n):
    return pytest.mark.skipif(torch.cuda.device_count() < n, reason=f"needs {n} GPUs")


class GridPipeline(pipe.Pipeline):
    """The pipeline on the real grid blocks of a model with fewer blocks than the released one
    (``full_model_execution`` takes the 50-block structure only)."""

    @contextmanager
    def execution(self, trace):
        from types import MethodType

        state = pm.ParallelState(self.layout.grid, self.sp)
        blocks = self.model.transformer_blocks
        for i in range(self.layout.stage.lo, self.layout.stage.hi):
            forward = pm._block_forward(i, self.layout.grid, self.sp, self.ops, state)
            blocks[i].forward = MethodType(forward, blocks[i])
        try:
            yield state
        finally:
            for block in blocks:
                block.__dict__.pop("forward", None)


def _grid_model():
    """Released dimensions, ``BLOCKS`` copies of block 0's weights, small seeded boundaries."""
    from rl_engine.validation.models import h3_block as hb
    from rl_engine.validation.models.h3_weights import load_h3_block_weights

    weights = load_h3_block_weights("cuda")
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = transformer.MiniMaxH3Transformer3DModel(
            num_layers=BLOCKS, num_refiner_layers=1, time_embed_hidden_dim=64
        )
    model = model.to(torch.bfloat16)
    for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"):
        getattr(model, name).to(torch.float32)
    model.rope.to(torch.float32)
    for block in model.transformer_blocks:
        for name, p in block.named_parameters():
            p.data = weights[
                f"{hb.BLOCK}.{name}"
            ].detach()  # one storage: the weights are only read
    for p in model.parameters():
        p.requires_grad_(False)
    model.enable_gradient_checkpointing()
    return model.eval()


def _grid_request(seed, text=20, frames_rows=200, audio=44):
    """A packed request of 264 rows (the block harness's "tiny" length)."""
    gen = torch.Generator().manual_seed(seed)
    seq = text + frames_rows + audio
    tags = torch.tensor([1] * text + [0] * frames_rows + [2] * audio)
    inputs = dict(
        hidden_states=torch.randn(1, frames_rows, 96, generator=gen),
        audio_hidden_states=torch.randn(1, audio, 32, generator=gen),
        encoder_hidden_states=torch.randn(1, text, 5120, generator=gen),
        timestep=torch.tensor([37.0, 500.0 + seed]),
        timestep_indices=(tags != 1).long(),
        token_tags=tags,
        position_ids=torch.arange(seq * 3).view(seq, 3) % 97,
        video_indices=torch.arange(text, text + frames_rows),
        audio_indices=torch.arange(text + frames_rows, seq),
        text_indices=torch.arange(text),
    )
    return {k: v.cuda() for k, v in inputs.items()}


def _grid_loss(n, video, audio):
    gen = torch.Generator().manual_seed(500 + n)
    gv = torch.randn(video.shape, generator=gen).to(video.device)
    ga = torch.randn(audio.shape, generator=gen).to(audio.device)
    return (video.float() * gv).sum() + (audio.float() * ga).sum()


def _grid_target(collective, dummy, *, pp, tp, cp, sp, cp_mode, sizes, schedule, trained_time):
    """This rank's stage against the canonical ``Grid(1, 1)`` run on one GPU."""
    import torch.distributed as dist

    from rl_engine.validation.models import h3_block as hb

    del collective, dummy
    torch.backends.cuda.matmul.allow_tf32 = False
    if trained_time:
        # the provider's final AdaLN gathers rows; that gather's CUDA backward does not repeat by
        # default, so without this the single-GPU run differs from itself in the time embedder
        torch.use_deterministic_algorithms(True, warn_only=True)
    model, ops = _grid_model(), hb.CandidateOps()
    for p in model.time_embedder.parameters():
        p.requires_grad_(trained_time)
    passes = [_grid_request(n) for n in range(max(3, pp + 1))]  # 1F1B needs one per stage
    singles = [dist.new_group([r]) for r in range(dist.get_world_size())]
    layout = pipe.make_layout(
        pp, tp, cp, total=BLOCKS, sizes=sizes, attention="h3_full", cp_mode=cp_mode
    )
    stage, grid = layout.stage, layout.grid

    def run(runner, cut):
        sums = {}
        inputs = with_grads(passes)

        def sink(index, name, grad):
            key = f"{index}.{name}"
            got = cut(name, grad)
            sums[key] = got if key not in sums else sums[key] + got

        outs, losses = runner.run(
            inputs,
            _grid_loss,
            trace=Trace(),
            schedule=schedule,
            configure=lambda state: setattr(state, "grad_sink", sink),
        )
        time_grads = [p.grad for p in model.time_embedder.parameters()]
        for p in model.time_embedder.parameters():
            p.grad = None
        return (
            outs,
            [float(x) for x in losses],
            [[p[k].grad for k in LEAVES] for p in inputs],
            (
                sums,
                time_grads,
            ),
        )

    alone = pipe.Layout(
        pipe.StageSpec(1, 0, 0, BLOCKS, BLOCKS),
        h3_tp.Grid(1, 1, 0, 0, attention="h3_full", group=singles[dist.get_rank()]),
        world=singles[dist.get_rank()],
    )
    ref = run(
        GridPipeline(model, alone, sp=False, strict_boundaries=False, ops=ops),
        lambda name, g: pm.shard_of(g, name, tp, grid.tp_rank),
    )
    staged = GridPipeline(model, layout, sp=sp, strict_boundaries=False, ops=ops)
    got = run(staged, lambda name, g: g)
    bad = []
    if schedule == "1f1b" and staged._groups(passes, schedule) != [(0, len(passes), "1f1b")]:
        bad.append("the passes did not run as 1F1B")
    for n in range(len(passes)):
        bad += [f"out{n}.{k}" for k in (0, 1) if not torch.equal(ref[0][n][k], got[0][n][k])]
        if stage.first:
            bad += [
                f"in{n}.{k}"
                for k, a, b in zip(LEAVES, ref[2][n], got[2][n])
                if b is None or not torch.equal(a, b)
            ]
    if ref[1] != got[1]:
        bad.append("losses")
    want = {k: v for k, v in ref[3][0].items() if stage.owns_block(int(k.split(".")[0]))}
    if set(want) != set(got[3][0]) or len(want) != (stage.hi - stage.lo) * len(pm.SHARDED_WEIGHTS):
        bad.append("weight shards: other keys")
    bad += [k for k in want if k in got[3][0] and not torch.equal(want[k], got[3][0][k])]
    if trained_time and stage.first:
        bad += [
            f"time_embedder.{i}"
            for i, (a, b) in enumerate(zip(ref[3][1], got[3][1]))
            if b is None or not torch.equal(a, b)
        ]
    return {"bad": bad[:8], "n": torch.tensor(len(bad))}


@pytest.mark.parametrize(
    ("pp", "tp", "cp", "sp", "mode", "sizes", "schedule", "trained_time"),
    [
        pytest.param(2, 1, 1, False, "ring", None, "1f1b", True, marks=gpus(2)),
        pytest.param(2, 1, 1, False, "ring", [1, 3], "serial", True, marks=gpus(2)),
        pytest.param(4, 1, 1, False, "ring", None, "gpipe", True, marks=gpus(4)),
        pytest.param(2, 2, 1, False, "ring", None, "1f1b", True, marks=gpus(4)),
        pytest.param(2, 2, 1, True, "ring", None, "serial", False, marks=gpus(4)),
        pytest.param(2, 1, 2, False, "ulysses", None, "1f1b", False, marks=gpus(4)),
        pytest.param(2, 2, 2, True, "ring", None, "1f1b", False, marks=gpus(8)),
        pytest.param(4, 2, 1, False, "ring", None, "1f1b", True, marks=gpus(8)),
        pytest.param(2, 4, 1, False, "ring", None, "gpipe", True, marks=gpus(8)),
    ],
)
def test_stages_on_the_grid_equal_the_single_gpu_run_bitwise(
    pp, tp, cp, sp, mode, sizes, schedule, trained_time, h3_block_weights_cuda
):
    """The accumulated passes: velocities, losses, input, weight-shard and time-embedder
    gradients of PP x TP x CP (x SP) equal the canonical single-GPU run bit for bit."""
    del h3_block_weights_cuda
    pytest.importorskip("triton")
    from rl_engine.backends.cuda.attention import h3_full_attention as fa
    from rl_engine.validation.distributed.h3_ws2 import run_world

    try:
        fa._extension()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    ranks = run_world(
        pp * tp * cp,
        _grid_target,
        {"dummy": torch.zeros(1)},
        deterministic_collective=False,
        pp=pp,
        tp=tp,
        cp=cp,
        sp=sp,
        cp_mode=mode,
        sizes=sizes,
        schedule=schedule,
        trained_time=trained_time,
    )
    for r in ranks:
        assert int(r["n"]) == 0, f"rank {r['rank']}: {r['bad']}"
