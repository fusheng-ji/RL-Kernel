# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the repeated-evidence fast path of ``FrameworkOperatorIntegration``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl_engine.integrations.common import runtime
from rl_engine.integrations.common.runtime import FrameworkOperatorIntegration
from rl_engine.runtime.plan import IntegrationPlan

pytestmark = pytest.mark.unit


class _Operator:
    backend_id = "rlkernel.test.attention"

    def __init__(self) -> None:
        self.provenance = {"runtime_platform": "cuda", "actual_backend": "rlkernel.test",
                           "nested": {"fallback": False}}
        self.result_provenance = {"split": {"num_splits": 1}}

    def __call__(self, value):
        return SimpleNamespace(value=value, provenance=dict(self.result_provenance))


def _integration(operator):
    return FrameworkOperatorIntegration(
        framework="megatron",
        target="training",
        plan=IntegrationPlan.from_case_ids(attention="R/R"),
        rl_kernel_operators={"attention": operator},
    )


def _count_scans(monkeypatch):
    calls = {"n": 0}
    original = runtime._contains_fallback

    def counting(value):
        calls["n"] += 1
        return original(value)

    monkeypatch.setattr(runtime, "_contains_fallback", counting)
    return calls


def test_repeated_identical_evidence_is_counted_without_rescanning(monkeypatch):
    operator = _Operator()
    integration = _integration(operator)
    scans = _count_scans(monkeypatch)
    for value in range(5):
        assert integration.execute("attention", lambda x: x, value).value == value
    assert scans["n"] > 0
    first_scans = scans["n"]
    integration.execute("attention", lambda x: x, 9)
    assert scans["n"] == first_scans
    readback = integration.readback()["operators"]["attention"]
    assert readback["call_count"] == 6
    assert readback["provenance"]["result"] == {"split": {"num_splits": 1}}


def test_changed_or_mutated_evidence_takes_the_full_path(monkeypatch):
    operator = _Operator()
    integration = _integration(operator)
    integration.execute("attention", lambda x: x, 0)
    scans = _count_scans(monkeypatch)
    operator.result_provenance = {"split": {"num_splits": 2}}
    integration.execute("attention", lambda x: x, 1)
    assert scans["n"] > 0
    scans["n"] = 0
    operator.provenance["nested"]["fallback"] = True  # in-place edit of the memoized dict
    with pytest.raises(RuntimeError, match="fallback"):
        integration.execute("attention", lambda x: x, 2)
    assert scans["n"] > 0


def test_direct_record_invalidates_the_memo():
    operator = _Operator()
    integration = _integration(operator)
    integration.execute("attention", lambda x: x, 0)
    integration.record_execution("attention", operator, execution_mode="compiled_cuda_graph")
    integration.execute("attention", lambda x: x, 1)
    readback = integration.readback()["operators"]["attention"]
    assert readback["execution_mode"] == "eager"
    assert readback["call_count"] == 3


def test_inferred_platform_is_never_memoized():
    operator = _Operator()
    del operator.provenance["runtime_platform"]
    integration = _integration(operator)
    integration.execute("attention", lambda x: x, 0)
    assert "attention" not in integration._execute_memo


def test_unsnapshottable_or_incomparable_evidence():
    class _Bad:
        def __eq__(self, other):
            raise RuntimeError("ambiguous")

        def __deepcopy__(self, memo):
            raise TypeError("no copy")

    assert runtime._evidence_snapshot(_Bad()) is runtime._UNSNAPSHOTTABLE
    assert runtime._same_evidence(_Bad(), {}) is False
    assert runtime._same_evidence({"a": [1]}, {"a": [1]}) is True
