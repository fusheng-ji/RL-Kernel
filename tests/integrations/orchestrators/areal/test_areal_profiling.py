# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the opt-in AReaL actor profiler."""

import json
import types

import pytest
import torch

from rl_engine.integrations.orchestrators.areal import profiling

pytestmark = pytest.mark.unit


def test_profile_steps_parsing(monkeypatch):
    assert profiling.profile_steps("") == frozenset()
    assert profiling.profile_steps(" 3, 1,3 ") == frozenset({1, 3})
    with pytest.raises(ValueError):
        profiling.profile_steps("-1")
    monkeypatch.delenv(profiling.PROFILE_STEPS_ENV, raising=False)
    assert not profiling.enabled()
    monkeypatch.setenv(profiling.PROFILE_STEPS_ENV, "2")
    assert profiling.enabled()


def test_ranged_wrapper_is_idempotent_and_transparent():
    def add(a, b=1):
        return a + b

    wrapped = profiling._ranged("add", add)
    assert profiling._ranged("add", wrapped) is wrapped
    assert wrapped(2, b=3) == 5
    assert wrapped.__name__ == "add"


def test_summarize_collects_ranges_on_cpu():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with torch.profiler.record_function(profiling.RANGE_PREFIX + "mlp"):
            torch.ones(8).sum()
    summary = profiling.summarize(prof, wall_seconds=1.5)
    assert summary["wall_s"] == 1.5
    assert summary["ranges"]["mlp"]["calls"] == 1
    assert summary["device_kernel_s"] == 0.0


def test_profiled_only_selected_versions(monkeypatch, tmp_path):
    monkeypatch.setenv(profiling.PROFILE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.profiler.ProfilerActivity, "CUDA", torch.profiler.ProfilerActivity.CPU
    )
    calls = []

    def forward_batch(self, value):
        calls.append(value)
        return value * 2

    wrapped = profiling._profiled("forward_batch", forward_batch, frozenset({1}))
    engine = types.SimpleNamespace(get_version=lambda: 0)
    assert wrapped(engine, 3) == 6
    assert list(tmp_path.iterdir()) == []
    engine.get_version = lambda: 1
    assert wrapped(engine, 4) == 8
    summary = json.loads((tmp_path / "forward_batch-v1-1.json").read_text())
    assert summary["kind"] == "forward_batch" and summary["version"] == 1
    assert (tmp_path / "forward_batch-v1-1.txt").exists()
    assert calls == [3, 4]
