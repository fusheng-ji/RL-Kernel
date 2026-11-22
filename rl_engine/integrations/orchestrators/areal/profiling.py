# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Opt-in ``torch.profiler`` capture of AReaL's Megatron actor steps.

Set ``RLK_AREAL_PROFILE_STEPS`` to a comma-separated list of actor versions
(``3`` or ``2,3``) and ``RLK_AREAL_PROFILE_DIR`` to an output directory. On
actor rank 0, every ``MegatronEngine.forward_batch`` (recomputed logp) and
``train_batch`` (PPO update) call made at a selected version is profiled, and
a JSON summary plus the ``key_averages`` table are written per call.

Coarse ``record_function`` ranges (``rlk::attention``, ``rlk::mlp``,
``rlk::rope``, ``rlk::lm_head``, ``rlk::det_collective``) are added around the
same Megatron entry points in both arms, so native and RL-Kernel profiles
aggregate into comparable buckets. Nothing is installed unless the steps
variable is set.
"""

from __future__ import annotations

import functools
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from rl_engine.integrations.orchestrators.areal.launch import PROFILE_DIR_ENV, PROFILE_STEPS_ENV

RANGE_PREFIX = "rlk::"
_ENGINE_MARKER = "__rl_kernel_areal_profiler__"
_RANGE_MARKER = "__rl_kernel_profile_range__"
_TOP_OPS = 40


def profile_steps(raw: str | None = None) -> frozenset[int]:
    value = os.getenv(PROFILE_STEPS_ENV, "") if raw is None else raw
    steps: set[int] = set()
    for item in value.replace(" ", "").split(","):
        if not item:
            continue
        step = int(item)
        if step < 0:
            raise ValueError(f"{PROFILE_STEPS_ENV} entries must be >= 0, got {step}")
        steps.add(step)
    return frozenset(steps)


def enabled() -> bool:
    return bool(profile_steps())


def _ranged(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(function, _RANGE_MARKER, None) is not None:
        return function
    label = RANGE_PREFIX + name

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with torch.profiler.record_function(label):
            return function(*args, **kwargs)

    setattr(wrapped, _RANGE_MARKER, label)
    return wrapped


def _range_attribute(owner: Any, attribute: str, name: str) -> None:
    function = getattr(owner, attribute, None)
    if callable(function):
        setattr(owner, attribute, _ranged(name, function))


def install_ranges(engine_module: ModuleType) -> None:
    """Wrap the Megatron entry points both arms share; call after strict patches."""

    import importlib

    attention = importlib.import_module("megatron.core.transformer.attention")
    _range_attribute(attention, "apply_rotary_pos_emb", "rope")
    dot_product = importlib.import_module("megatron.core.transformer.dot_product_attention")
    _range_attribute(dot_product.DotProductAttention, "forward", "attention")
    try:
        te_ext = importlib.import_module("megatron.core.extensions.transformer_engine")
    except ImportError:
        te_ext = None
    if te_ext is not None:
        for name in ("TEDotProductAttention", "TEDotProductAttentionWithCP"):
            cls = getattr(te_ext, name, None)
            if cls is not None and "forward" in vars(cls):
                _range_attribute(cls, "forward", "attention")
    mlp = importlib.import_module("megatron.core.transformer.mlp")
    _range_attribute(mlp.MLP, "forward", "mlp")
    _range_attribute(engine_module, "chunked_lm_head_logprobs_entropy", "lm_head")
    try:
        collectives = importlib.import_module("rl_engine.distributed.algorithms.collectives")
    except ImportError:
        collectives = None
    if collectives is not None:
        cls = collectives.DeterministicCollective
        for method in ("all_reduce", "all_gather", "all_gather_many", "reduce_scatter"):
            _range_attribute(cls, method, f"det_collective.{method}")


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def summarize(profiler: Any, *, wall_seconds: float, averages: Any = None) -> dict[str, Any]:
    """Device/host totals, coarse ranges and the top device-time operators."""

    if averages is None:
        averages = profiler.key_averages()
    ranges: dict[str, dict[str, float]] = {}
    ops: list[dict[str, Any]] = []
    device_total_us = 0.0
    for event in averages:
        self_device = float(getattr(event, "self_device_time_total", 0.0))
        device_total_us += self_device
        if event.key.startswith(RANGE_PREFIX):
            ranges[event.key[len(RANGE_PREFIX) :]] = {
                "calls": int(event.count),
                "cpu_s": float(event.cpu_time_total) / 1e6,
                "device_s": float(getattr(event, "device_time_total", 0.0)) / 1e6,
            }
        elif self_device > 0:
            ops.append(
                {
                    "name": event.key,
                    "calls": int(event.count),
                    "self_device_s": self_device / 1e6,
                }
            )
    ops.sort(key=lambda item: item["self_device_s"], reverse=True)
    return {
        "wall_s": wall_seconds,
        "device_kernel_s": device_total_us / 1e6,
        "ranges": dict(sorted(ranges.items())),
        "top_device_ops": ops[:_TOP_OPS],
    }


def _profiled(kind: str, original: Callable[..., Any], steps: frozenset[int]) -> Callable[..., Any]:
    counter = {"calls": 0}

    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        version = int(self.get_version())
        if version not in steps or _rank() != 0:
            return original(self, *args, **kwargs)
        counter["calls"] += 1
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.profiler.profile(activities=activities) as profiler:
            result = original(self, *args, **kwargs)
            torch.cuda.synchronize()
        wall = time.perf_counter() - start
        directory = Path(os.getenv(PROFILE_DIR_ENV) or "rlk-profile")
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{kind}-v{version}-{counter['calls']}"
        # Aggregating a full PPO step takes minutes; do it once for both outputs.
        averages = profiler.key_averages()
        summary = {
            "kind": kind,
            "version": version,
            **summarize(profiler, wall_seconds=wall, averages=averages),
        }
        (directory / f"{stem}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        table = averages.table(sort_by="self_device_time_total", row_limit=60)
        (directory / f"{stem}.txt").write_text(table, encoding="utf-8")
        print(
            f"[RL-Kernel][profile] {stem}: wall={wall:.2f}s "
            f"device_kernels={summary['device_kernel_s']:.2f}s -> {directory}",
            flush=True,
        )
        return result

    return wrapped


def install_engine_profiler(engine_module: ModuleType) -> None:
    steps = profile_steps()
    if not steps or getattr(engine_module, _ENGINE_MARKER, False):
        return
    install_ranges(engine_module)
    engine_cls = engine_module.MegatronEngine
    engine_cls.forward_batch = _profiled("forward_batch", engine_cls.forward_batch, steps)
    engine_cls.train_batch = _profiled("train_batch", engine_cls.train_batch, steps)
    setattr(engine_module, _ENGINE_MARKER, True)
    print(f"[RL-Kernel][profile] actor profiler armed for versions {sorted(steps)}", flush=True)


__all__ = [
    "PROFILE_DIR_ENV",
    "PROFILE_STEPS_ENV",
    "enabled",
    "install_engine_profiler",
    "install_ranges",
    "profile_steps",
    "summarize",
]
