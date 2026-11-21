# SPDX-License-Identifier: Apache-2.0
"""The external LM-head mode must not touch LinearCrossEntropyModule."""

import pytest

from rl_engine.integrations.common.state import clear_active_integration
from rl_engine.integrations.engines.train.megatron import runtime
from rl_engine.runtime.plan import IntegrationPlan

pytestmark = pytest.mark.unit


class _Attention:
    def forward(self, *args, **kwargs):
        return "attention"


class _FFN:
    def forward(self, *args, **kwargs):
        return "ffn"


@pytest.fixture
def clean_integration():
    clear_active_integration("megatron")
    yield
    clear_active_integration("megatron")


def test_unknown_output_layer_mode_is_rejected(clean_integration):
    plan = IntegrationPlan.from_case_ids(attention="P/P", ffn="P/P", logp="R/R")
    with pytest.raises(ValueError, match="output_layer_mode"):
        runtime.install_megatron_integration(plan, output_layer_mode="bogus")


def test_external_mode_skips_linear_cross_entropy_patch(clean_integration, monkeypatch):
    patched = []
    monkeypatch.setattr(runtime, "_patch_strict_logp_output_layer", lambda: patched.append(1))
    monkeypatch.setattr(runtime, "_patch_layer_alignment_diagnostics", lambda: None)
    plan = IntegrationPlan.from_case_ids(attention="P/P", ffn="P/P", logp="R/R")
    integration = runtime.install_megatron_integration(
        plan,
        attention_classes=[_Attention],
        ffn_classes=[_FFN],
        output_layer_mode="external",
    )
    assert patched == []
    assert "logp" not in integration.readback()["installed_hooks"]
    with pytest.raises(RuntimeError, match="another LM head"):
        runtime.install_megatron_integration(plan, output_layer_mode="linear_cross_entropy")
