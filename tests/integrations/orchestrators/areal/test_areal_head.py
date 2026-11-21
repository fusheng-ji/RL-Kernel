# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the strict AReaL LM-head call contract."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.orchestrators.areal.head import validate_head_call

pytestmark = pytest.mark.unit


def _layer(**overrides):
    values = dict(sequence_parallel=False, gather_output=False, bias=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def _tensors(rows=6, hidden=8, vocab=16, dtype=torch.bfloat16):
    return (
        torch.zeros(rows, 1, hidden, dtype=dtype),
        torch.zeros(vocab, hidden, dtype=dtype),
        torch.zeros(rows, dtype=torch.int64),
    )


def test_supported_call_passes():
    hidden, weight, labels = _tensors()
    validate_head_call(_layer(), hidden, weight, labels, temperature=1.0, logit_scale=1.0)


@pytest.mark.parametrize(
    ("layer", "kwargs", "fragment"),
    [
        (_layer(), {"temperature": 0.7}, "temperature"),
        (_layer(), {"logit_scale": 2.0}, "MuP"),
        (_layer(sequence_parallel=True), {}, "sequence_parallel"),
        (_layer(gather_output=True), {}, "gather_output"),
        (_layer(bias=torch.zeros(16)), {}, "bias-free"),
    ],
)
def test_unsupported_calls_raise(layer, kwargs, fragment):
    hidden, weight, labels = _tensors()
    call = {"temperature": 1.0, "logit_scale": 1.0, **kwargs}
    with pytest.raises(ValueError, match=fragment):
        validate_head_call(layer, hidden, weight, labels, **call)


def test_dtype_and_row_alignment_are_checked():
    hidden, weight, labels = _tensors(dtype=torch.float32)
    with pytest.raises(ValueError, match="BF16"):
        validate_head_call(_layer(), hidden, weight, labels, temperature=1.0, logit_scale=1.0)
    hidden, weight, _ = _tensors()
    with pytest.raises(ValueError, match="match labels"):
        validate_head_call(_layer(), hidden, weight, torch.zeros(5, dtype=torch.int64),
                           temperature=1.0, logit_scale=1.0)
