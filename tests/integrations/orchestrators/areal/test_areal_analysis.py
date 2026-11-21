# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the verification verdict logic."""

import json

import pytest

from rl_engine.integrations.orchestrators.areal import analysis

pytestmark = pytest.mark.unit

R3 = "compute_logp/r3/rollout_train_logp_abs_diff"
OURS = "compute_logp/consistency"


def _write_steps(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for step, stats in rows:
            handle.write(json.dumps({"global_step": step, "stats": stats}) + "\n")


def _strict_stats(mismatch=0.0, same=1000.0, same_mismatch=0.0):
    return {
        f"{R3}/max": 0.0 if not mismatch else 0.5,
        f"{R3}/avg": 0.0,
        f"{OURS}/n_valid_tokens": 4000.0,
        f"{OURS}/mismatch_tokens": mismatch,
        f"{OURS}/nonfinite_tokens": 0.0,
        f"{OURS}/n_same_version_tokens": same,
        f"{OURS}/same_version_mismatch_tokens": same_mismatch,
    }


def _full_provenance(tmp_path):
    readback = tmp_path / "readback"
    readback.mkdir(exist_ok=True)
    for framework in ("megatron", "vllm"):
        payload = {
            "framework": framework,
            "fallbacks": [],
            "operators": {m: {"call_count": 3} for m in ("attention", "ffn", "logp")},
        }
        (readback / f"{framework}-x-1.json").write_text(json.dumps(payload), encoding="utf-8")
    for role in ("training", "rollout"):
        (readback / f"cuda-rope-{role}-1.json").write_text(
            json.dumps({"module": "rope", "role": role}), encoding="utf-8"
        )
    return analysis.load_provenance(readback)


def test_sync_pass_requires_zero_mismatch_every_step_and_provenance(tmp_path):
    path = tmp_path / "stats.jsonl"
    _write_steps(path, [(1, _strict_stats()), (2, _strict_stats())])
    steps = analysis.load_steps(path)
    verdict = analysis.judge_arm("rlkernel", "sync", steps, expected_steps=2,
                                 provenance=_full_provenance(tmp_path))
    assert verdict.verdict == analysis.PASS, verdict.reasons

    _write_steps(path, [(1, _strict_stats()), (2, _strict_stats(mismatch=3.0))])
    verdict = analysis.judge_arm("rlkernel", "sync", analysis.load_steps(path),
                                 expected_steps=2, provenance=_full_provenance(tmp_path))
    assert verdict.verdict == analysis.FAIL
    assert any("3 mismatching" in reason for reason in verdict.reasons)


def test_sync_fails_without_route_evidence(tmp_path):
    path = tmp_path / "stats.jsonl"
    _write_steps(path, [(1, _strict_stats())])
    verdict = analysis.judge_arm("rlkernel", "sync", analysis.load_steps(path),
                                 expected_steps=1, provenance=analysis.Provenance())
    assert verdict.verdict == analysis.FAIL
    assert any("not proven" in reason for reason in verdict.reasons)


def test_async_judges_only_same_version_tokens(tmp_path):
    path = tmp_path / "stats.jsonl"
    # All-token mismatches are expected under staleness; same-version must be zero.
    _write_steps(path, [(1, _strict_stats(mismatch=50.0, same=6000.0)),
                        (2, _strict_stats(mismatch=70.0, same=6000.0))])
    provenance = _full_provenance(tmp_path)
    verdict = analysis.judge_arm("rlkernel", "async", analysis.load_steps(path),
                                 expected_steps=2, provenance=provenance)
    assert verdict.verdict == analysis.PASS, verdict.reasons

    _write_steps(path, [(1, _strict_stats(same=100.0))])
    verdict = analysis.judge_arm("rlkernel", "async", analysis.load_steps(path),
                                 expected_steps=1, provenance=provenance)
    assert verdict.verdict == analysis.INCONCLUSIVE

    _write_steps(path, [(1, _strict_stats(same=6000.0, same_mismatch=1.0))])
    verdict = analysis.judge_arm("rlkernel", "async", analysis.load_steps(path),
                                 expected_steps=1, provenance=provenance)
    assert verdict.verdict == analysis.FAIL


def test_native_arm_is_report_only_but_must_not_run_rl_kernel(tmp_path):
    path = tmp_path / "stats.jsonl"
    _write_steps(path, [(1, {f"{R3}/max": 1.25, f"{R3}/avg": 0.01})])
    steps = analysis.load_steps(path)
    verdict = analysis.judge_arm("native", "sync", steps, expected_steps=1,
                                 provenance=analysis.Provenance())
    assert verdict.verdict == analysis.REPORT_ONLY
    assert steps[0].abs_diff_max == 1.25
    verdict = analysis.judge_arm("native", "sync", steps, expected_steps=1,
                                 provenance=_full_provenance(tmp_path))
    assert verdict.verdict == analysis.FAIL


def test_lookup_accepts_prefixed_keys_and_report_renders(tmp_path):
    assert analysis.lookup({f"ppo_actor/{R3}/max": 2.0}, f"{R3}/max") == 2.0
    assert analysis.lookup({}, "missing") is None
    path = tmp_path / "stats.jsonl"
    _write_steps(path, [(1, _strict_stats())])
    steps = analysis.load_steps(path)
    verdict = analysis.judge_arm("rlkernel", "async", steps, expected_steps=1,
                                 provenance=_full_provenance(tmp_path))
    report = analysis.render_report("async", {"rlkernel": (steps, verdict)})
    assert "same-version tokens" in report
