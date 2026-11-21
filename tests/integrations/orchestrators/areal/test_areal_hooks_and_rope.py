# SPDX-License-Identifier: Apache-2.0
"""CPU checks for lazy patch hooks and the shared CUDA RoPE layout glue."""

import sys

import pytest
import torch

from rl_engine.integrations.common import cuda_rope
from rl_engine.integrations.orchestrators.areal import hooks

pytestmark = pytest.mark.unit


def test_when_imported_runs_after_import(tmp_path, monkeypatch):
    package = tmp_path / "rlk_fake_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    hooks.when_imported("rlk_fake_pkg.target", lambda module: seen.append(module.VALUE))
    assert seen == []
    import rlk_fake_pkg.target  # noqa: F401

    assert seen == [1]
    hooks.when_imported("rlk_fake_pkg.target", lambda module: seen.append(module.VALUE + 1))
    assert seen == [1, 2]  # already imported -> immediate
    for name in ("rlk_fake_pkg.target", "rlk_fake_pkg"):
        sys.modules.pop(name, None)


def test_token_major_rope_round_trips_layouts():
    calls = []

    def operator(heads, positions, *, theta):
        calls.append((tuple(heads.shape), positions.tolist(), theta))
        return heads * 2

    value = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 8)
    out = cuda_rope.token_major_rope(value, torch.tensor([5, 6, 7]), 4, operator)
    assert torch.equal(out, value * 2)
    assert calls[0] == ((2, 3, 4), [5, 6, 7], cuda_rope.QWEN3_THETA)
    three_d = value.reshape(3, 2, 4)
    assert cuda_rope.token_major_rope(three_d, torch.tensor([0, 1, 2]), 4, operator).shape == (
        3, 2, 4
    )
    with pytest.raises(ValueError, match="position count"):
        cuda_rope.token_major_rope(value, torch.tensor([0, 1]), 4, operator)


def test_logical_positions_read_channel_zero_of_reset_tables():
    # A packed table that resets per sequence: positions 0,1,2 then 0,1.
    channel = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0])
    freqs = channel.reshape(5, 1, 1, 1).expand(5, 1, 1, 8).contiguous()
    indices = torch.tensor([0, 1, 2, 3, 4])
    assert cuda_rope.logical_positions(freqs, indices).tolist() == [0, 1, 2, 0, 1]
    broken = freqs.clone()
    broken[1, 0, 0, 0] = 0.5
    with pytest.raises(RuntimeError, match="integer-position"):
        cuda_rope.logical_positions(broken, indices)


def test_cuda_rope_enable_flag(monkeypatch):
    monkeypatch.setenv(cuda_rope.ENV_FLAG, "1")
    assert cuda_rope.enabled() is (torch.version.hip is None)
    monkeypatch.setenv(cuda_rope.ENV_FLAG, "0")
    assert cuda_rope.enabled() is False


def test_patches_install_observe_mode_only_wraps_actor(monkeypatch):
    from rl_engine.integrations.orchestrators.areal import patches

    registered = []
    monkeypatch.setattr(patches, "when_imported", lambda name, cb: registered.append(name))
    monkeypatch.delenv("RL_KERNEL_AREAL_MODE", raising=False)
    assert patches.install() is None
    assert registered == []
    assert patches.install("observe") == "observe"
    assert registered == [patches.ACTOR_MODULE]
    registered.clear()
    assert patches.install("strict") == "strict"
    assert registered == [patches.ACTOR_MODULE, patches.ENGINE_MODULE]
    monkeypatch.delenv("RL_KERNEL_AREAL_MODE", raising=False)
