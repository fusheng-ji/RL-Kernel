# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Same-version train/rollout logp observation for AReaL.

AReaL already logs ``compute_logp/r3/rollout_train_logp_abs_diff`` for every
recomputed batch, but in asynchronous training that metric mixes engine noise
with genuine policy staleness. Only tokens whose generation version equals the
actor version at recomputation compare two log-probabilities from identical
weights, so only those must agree bitwise.

This module adds exact counts and breakdowns next to AReaL's own metric. It is
observation only: it never changes loss, masks, advantages or weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

STATS_SCOPE = ("compute_logp", "consistency")
_ACTOR_MARKER = "__rl_kernel_original_r3_logp_stats__"
_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"[RL-Kernel][areal] WARNING {message}", flush=True)


def _rows(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(0) if value.ndim == 1 else value


def _next_token_aligned(value: torch.Tensor, fill: Any) -> torch.Tensor:
    """AReaL stores per-token values; training logp at ``i`` predicts token ``i+1``."""

    rolled = torch.roll(value, shifts=-1, dims=-1)
    if rolled.shape[-1] > 0:
        rolled[..., -1] = fill
    return rolled


def consistency_masks(
    *,
    train_logp: torch.Tensor,
    rollout_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> dict[str, torch.Tensor]:
    """Return aligned difference and token masks.

    All inputs use AReaL's raw (unrolled) per-token layout except ``train_logp``,
    which is already next-token aligned by the engine.
    """

    device = train_logp.device
    rollout = _next_token_aligned(rollout_logp.to(device), 0.0)
    valid = _next_token_aligned(loss_mask.to(device).bool(), False)
    if not (train_logp.shape == rollout.shape == valid.shape):
        raise ValueError(
            "train/rollout/mask shapes differ: "
            f"{tuple(train_logp.shape)}, {tuple(rollout.shape)}, {tuple(valid.shape)}"
        )
    delta = train_logp.detach().float() - rollout.detach().float()
    nonfinite = valid & ~torch.isfinite(delta)
    mismatch = valid & (delta != 0)
    masks: dict[str, torch.Tensor] = {
        "delta": delta,
        "valid": valid,
        "mismatch": mismatch,
        "nonfinite": nonfinite,
    }
    if versions is None or current_version is None:
        return masks
    aligned_versions = _next_token_aligned(versions.to(device).long(), -1)
    if aligned_versions.shape != valid.shape:
        raise ValueError("versions shape differs from the token layout")
    generated = valid & (aligned_versions >= 0)
    same = generated & (aligned_versions == int(current_version))
    stale = generated & (aligned_versions < int(current_version))
    future = generated & (aligned_versions > int(current_version))
    rows_generated = _rows(generated)
    rows_versions = _rows(aligned_versions)
    high = torch.where(rows_generated, rows_versions, torch.full_like(rows_versions, -(2**62)))
    low = torch.where(rows_generated, rows_versions, torch.full_like(rows_versions, 2**62))
    mixed_rows = rows_generated.any(dim=-1) & (high.amax(dim=-1) > low.amin(dim=-1))
    mixed = (rows_generated & mixed_rows.unsqueeze(-1)).reshape(generated.shape)
    masks.update(
        {
            "same_version": same,
            "stale": stale,
            "future": future,
            "mixed_seq_same_version": same & mixed,
        }
    )
    return masks


def summarize(masks: Mapping[str, torch.Tensor]) -> dict[str, float | None]:
    """Plain-Python summary used by tests and offline inspection."""

    delta = masks["delta"].abs()
    out: dict[str, float | None] = {}
    for name in ("valid", "same_version", "stale", "future", "mixed_seq_same_version"):
        if name not in masks:
            continue
        mask = masks[name]
        count = int(mask.sum().item())
        out[f"n_{name}_tokens"] = float(count)
        out[f"{name}_mismatch_tokens"] = float((mask & masks["mismatch"]).sum().item())
        out[f"{name}_logp_abs_diff_max"] = float(delta[mask].max().item()) if count else None
    out["nonfinite_tokens"] = float(masks["nonfinite"].sum().item())
    return out


def record(masks: Mapping[str, torch.Tensor]) -> None:
    """Export exact counts and breakdowns through AReaL's stats tracker."""

    from areal.utils import stats_tracker
    from areal.utils.stats_tracker import ReduceType

    abs_delta = masks["delta"].abs()
    ones = torch.ones_like(abs_delta)
    mismatch = masks["mismatch"].float()
    with stats_tracker.scope(STATS_SCOPE[0]):
        with stats_tracker.scope(STATS_SCOPE[1]):
            stats_tracker.stat_compact(
                masks["valid"],
                ReduceType.SUM,
                n_valid_tokens=ones,
                mismatch_tokens=mismatch,
                nonfinite_tokens=masks["nonfinite"].float(),
            )
            for name in ("same_version", "stale", "mixed_seq_same_version"):
                mask = masks.get(name)
                if mask is None:
                    continue
                # Record every key on every rank, even for empty masks: AReaL
                # reduces keys collectively across data-parallel workers.
                stats_tracker.stat_compact(
                    mask,
                    ReduceType.SUM,
                    **{f"n_{name}_tokens": ones, f"{name}_mismatch_tokens": mismatch},
                )
                stats_tracker.stat_compact(
                    mask,
                    ReduceType.AVG_MIN_MAX,
                    **{f"{name}_logp_abs_diff": abs_delta},
                )


def install_actor_stats(actor_module: Any) -> None:
    """Wrap ``PPOActor._log_r3_rollout_train_logp_stats`` to add the breakdown."""

    actor_cls = actor_module.PPOActor
    if hasattr(actor_cls, _ACTOR_MARKER):
        return
    original = actor_cls._log_r3_rollout_train_logp_stats

    def wrapped(self: Any, *, train_logp: Any, data: dict[str, Any], r3_enabled: bool) -> None:
        original(self, train_logp=train_logp, data=data, r3_enabled=r3_enabled)
        if not torch.is_tensor(train_logp):
            return
        rollout = data.get("logprobs")
        loss_mask = data.get("loss_mask")
        if not torch.is_tensor(rollout) or not torch.is_tensor(loss_mask):
            return
        versions = data.get("versions")
        current = None
        engine = getattr(self, "engine", None)
        if engine is not None and callable(getattr(engine, "get_version", None)):
            current = int(engine.get_version())
        try:
            masks = consistency_masks(
                train_logp=train_logp,
                rollout_logp=rollout,
                loss_mask=loss_mask,
                versions=versions if torch.is_tensor(versions) else None,
                current_version=current,
            )
        except ValueError as error:
            # Mirror AReaL: an unexpected layout skips the metric, never the step.
            _warn_once(f"RL-Kernel consistency metrics skipped: {error}")
            return
        record(masks)

    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    setattr(actor_cls, _ACTOR_MARKER, original)
    actor_cls._log_r3_rollout_train_logp_stats = wrapped


__all__ = ["STATS_SCOPE", "consistency_masks", "install_actor_stats", "record", "summarize"]
