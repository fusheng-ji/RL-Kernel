# SPDX-License-Identifier: Apache-2.0
"""CPU checks for same-version train/rollout logp observation."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.orchestrators.areal import stats

pytestmark = pytest.mark.unit


def _batch():
    # Two sequences of six tokens. Prompt tokens have loss_mask 0 / version -1.
    # Row 0 was generated at version 3 then resumed at version 4 (mixed);
    # row 1 entirely at version 4.
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]])
    versions = torch.tensor([[-1, -1, 3, 3, 4, 4], [-1, 4, 4, 4, 4, -1]])
    rollout = torch.tensor(
        [[0.0, 0.0, -1.0, -2.0, -3.0, -4.0], [0.0, -0.5, -1.5, -2.5, -3.5, 0.0]]
    )
    # Training logp is next-token aligned: position i scores token i+1.
    train = torch.roll(rollout, shifts=-1, dims=-1).clone()
    return train, rollout, loss_mask, versions


def test_identical_logp_has_no_mismatch_and_correct_partitions():
    train, rollout, loss_mask, versions = _batch()
    masks = stats.consistency_masks(
        train_logp=train, rollout_logp=rollout, loss_mask=loss_mask,
        versions=versions, current_version=4,
    )
    summary = stats.summarize(masks)
    # Valid next-token targets: row 0 -> tokens 2..5 (versions 3,3,4,4),
    # row 1 -> tokens 1..4 (all version 4).
    assert summary["n_valid_tokens"] == 8.0
    assert summary["valid_mismatch_tokens"] == 0.0
    assert summary["n_same_version_tokens"] == 2.0 + 4.0
    assert summary["n_stale_tokens"] == 2.0
    assert summary["n_mixed_seq_same_version_tokens"] == 2.0
    assert summary["nonfinite_tokens"] == 0.0


def test_stale_difference_is_not_a_same_version_mismatch():
    train, rollout, loss_mask, versions = _batch()
    train[0, 1] += 0.25  # predicts token 2, generated at stale version 3
    masks = stats.consistency_masks(
        train_logp=train, rollout_logp=rollout, loss_mask=loss_mask,
        versions=versions, current_version=4,
    )
    summary = stats.summarize(masks)
    assert summary["valid_mismatch_tokens"] == 1.0
    assert summary["same_version_mismatch_tokens"] == 0.0
    assert summary["stale_mismatch_tokens"] == 1.0


def test_same_version_difference_is_counted_exactly():
    train, rollout, loss_mask, versions = _batch()
    train[1, 2] = torch.nextafter(train[1, 2], torch.tensor(0.0))
    masks = stats.consistency_masks(
        train_logp=train, rollout_logp=rollout, loss_mask=loss_mask,
        versions=versions, current_version=4,
    )
    summary = stats.summarize(masks)
    assert summary["same_version_mismatch_tokens"] == 1.0
    assert summary["same_version_logp_abs_diff_max"] > 0.0


def test_missing_versions_still_reports_all_token_counts():
    train, rollout, loss_mask, _ = _batch()
    masks = stats.consistency_masks(
        train_logp=train, rollout_logp=rollout, loss_mask=loss_mask,
        versions=None, current_version=None,
    )
    assert "same_version" not in masks
    assert stats.summarize(masks)["valid_mismatch_tokens"] == 0.0


def test_shape_mismatch_raises():
    train, rollout, loss_mask, versions = _batch()
    with pytest.raises(ValueError, match="shapes differ"):
        stats.consistency_masks(
            train_logp=train[:, :-1], rollout_logp=rollout, loss_mask=loss_mask,
            versions=versions, current_version=4,
        )


def test_actor_wrapper_records_through_stats_tracker(monkeypatch):
    recorded = []

    class _Scope:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    tracker = types.ModuleType("areal.utils.stats_tracker")
    tracker.ReduceType = SimpleNamespace(SUM="SUM", AVG_MIN_MAX="AVG_MIN_MAX")
    tracker.scope = _Scope
    tracker.stat_compact = lambda mask, reduce, **values: recorded.append(
        (reduce, sorted(values), int(mask.sum()))
    )
    utils = types.ModuleType("areal.utils")
    utils.stats_tracker = tracker
    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.utils", utils)
    monkeypatch.setitem(sys.modules, "areal.utils.stats_tracker", tracker)

    calls = []

    class PPOActor:
        def __init__(self):
            self.engine = SimpleNamespace(get_version=lambda: 4)

        def _log_r3_rollout_train_logp_stats(self, *, train_logp, data, r3_enabled):
            calls.append(r3_enabled)

    actor_module = types.ModuleType("fake_actor")
    actor_module.PPOActor = PPOActor
    stats.install_actor_stats(actor_module)
    stats.install_actor_stats(actor_module)  # idempotent

    train, rollout, loss_mask, versions = _batch()
    PPOActor()._log_r3_rollout_train_logp_stats(
        train_logp=train,
        data={"logprobs": rollout, "loss_mask": loss_mask, "versions": versions},
        r3_enabled=False,
    )
    assert calls == [False]
    keys = {key for _, names, _ in recorded for key in names}
    assert {"n_valid_tokens", "mismatch_tokens", "n_same_version_tokens",
            "same_version_mismatch_tokens", "same_version_logp_abs_diff"} <= keys

    # A layout the metric cannot align must skip the metric, not the step.
    recorded.clear()
    PPOActor()._log_r3_rollout_train_logp_stats(
        train_logp=train[:, :-1],
        data={"logprobs": rollout, "loss_mask": loss_mask, "versions": versions},
        r3_enabled=False,
    )
    assert recorded == []
