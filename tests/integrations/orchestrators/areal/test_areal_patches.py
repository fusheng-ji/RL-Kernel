# SPDX-License-Identifier: Apache-2.0
"""CPU checks for optional native RoPE fusion patches."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import pytest

from rl_engine.integrations.orchestrators.areal import contract, launch, patches

pytestmark = pytest.mark.unit


@dataclass
class _FakeTransformerConfig:
    apply_rope_fusion: bool = False

    def __post_init__(self) -> None:
        pass


def test_enable_native_rope_fusion_sets_apply_rope_fusion():
    module = ModuleType("transformer_config")
    module.TransformerConfig = _FakeTransformerConfig
    patches._enable_native_rope_fusion(module)
    patches._enable_native_rope_fusion(module)
    config = module.TransformerConfig()
    assert config.apply_rope_fusion is True


def test_install_rejects_strict_with_native_rope_fusion(monkeypatch):
    monkeypatch.setenv(launch.NATIVE_ROPE_FUSION_ENV, "1")
    monkeypatch.setenv(contract.MODE_ENV, "strict")
    with pytest.raises(ValueError, match="native RoPE fusion cannot"):
        patches.install("strict")


def test_native_rope_fusion_enabled_parses_env(monkeypatch):
    monkeypatch.delenv(launch.NATIVE_ROPE_FUSION_ENV, raising=False)
    assert not patches.native_rope_fusion_enabled()
    monkeypatch.setenv(launch.NATIVE_ROPE_FUSION_ENV, "1")
    assert patches.native_rope_fusion_enabled()