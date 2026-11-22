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


def _packed_table(lengths, *, exact):
    from rl_engine.integrations.engines.train.megatron.runtime import (
        _strict_rocm_rope_positions,
    )

    tokens = sum(lengths)
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    rows = tokens if exact else max(lengths)
    if exact:
        channel = torch.cat([torch.arange(n, dtype=torch.float32) for n in lengths])
    else:
        channel = torch.arange(rows, dtype=torch.float32)
    freqs = channel.reshape(rows, 1, 1, 1).expand(rows, 1, 1, 8).contiguous()
    tensor = torch.empty(tokens, 2, 8)
    expected = cuda_rope.logical_positions(
        freqs, _strict_rocm_rope_positions(tensor, freqs, cu_seqlens, None)
    )
    return tensor, freqs, cu_seqlens, expected


@pytest.mark.parametrize("exact", [True, False])
def test_packed_cp1_indices_match_host_builder(exact):
    for lengths in ([5], [3, 1, 7], [2, 9, 4, 4, 1, 6]):
        tensor, freqs, cu_seqlens, expected = _packed_table(lengths, exact=exact)
        indices = cuda_rope.packed_cp1_indices(cu_seqlens, tensor.size(0), freqs.size(0))
        assert torch.equal(cuda_rope.logical_positions(freqs, indices), expected)
    with pytest.raises(RuntimeError, match="invalid cu_seqlens"):
        cuda_rope.packed_cp1_indices(torch.tensor([0, 3, 3]), 3, 2)
    with pytest.raises(RuntimeError, match="invalid cu_seqlens"):
        cuda_rope.packed_cp1_indices(torch.tensor([0, 3]), 4, 2)


def test_packed_position_cache_reuses_one_layout_only():
    calls = []

    def fallback(tensor, freqs, cu_seqlens, cp_group):
        calls.append(cp_group)
        return torch.zeros(tensor.size(0), dtype=torch.int64)

    cache = cuda_rope.PackedPositionCache(fallback)
    tensor, freqs, cu_seqlens, expected = _packed_table([3, 4], exact=False)
    first = cache(tensor, freqs, cu_seqlens, None)
    assert torch.equal(first, expected)
    assert cache(tensor, freqs, cu_seqlens, None) is first
    assert cache(tensor, freqs, cu_seqlens.clone(), None) is not first
    other = cu_seqlens.clone()
    cache(tensor, freqs, other, None)
    other[1] = 4  # in-place edit bumps the version
    assert torch.equal(cache(tensor, freqs, other, None), torch.tensor([0, 1, 2, 3, 0, 1, 2]))
    assert calls == []

    class _Group:
        def size(self):
            return 2

    cache(tensor, freqs, cu_seqlens, _Group())
    cache(torch.empty(7, 1, 2, 8), freqs, None, None)
    assert len(calls) == 2


def test_shared_rope_tables_are_reused_and_bitwise_equal():
    from rl_engine.backends.cuda.rope import rope

    positions = torch.tensor([0, 1, 2, 0, 1, 7])
    half, theta = 4, cuda_rope.QWEN3_THETA
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = positions.float().reshape(-1, 1) * inv_freq
    plain = rope._build_cos_sin(positions, half, theta, positions.device)
    assert rope._build_cos_sin(positions, half, theta, positions.device)[0] is not plain[0]
    assert torch.equal(plain[0], freqs.cos()) and torch.equal(plain[1], freqs.sin())
    shared = rope.share_rope_tables(positions.view(-1))
    first = rope._build_cos_sin(shared, half, theta, shared.device)
    assert rope._build_cos_sin(shared, half, theta, shared.device) is first
    assert torch.equal(first[0], plain[0]) and torch.equal(first[1], plain[1])
    assert not hasattr(positions, "_rlk_rope_cos_sin")


def test_identity_temperature_skips_only_scalar_one():
    from rl_engine.integrations.common.linear_logp import _is_identity_scalar_temperature

    assert _is_identity_scalar_temperature(None)
    assert _is_identity_scalar_temperature(1.0) and _is_identity_scalar_temperature(1)
    assert not _is_identity_scalar_temperature(0.7)
    assert not _is_identity_scalar_temperature(torch.tensor(1.0))
