# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Turn per-step AReaL statistics and RL-Kernel readbacks into a verdict.

Inputs are the JSONL written by ``run_trainer`` and the JSON readbacks written
by RL-Kernel integrations. Pure Python, no AReaL or torch import.

Primary evidence is AReaL's own ``compute_logp/r3/rollout_train_logp_abs_diff``
metric. The adapter's exact counts (``compute_logp/consistency/*``) refine it:
mismatch *counts* instead of a maximum, and the same-version subset that
asynchronous training needs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
REPORT_ONLY = "REPORT_ONLY"

AREAL_ABS_DIFF = "compute_logp/r3/rollout_train_logp_abs_diff"
OURS = "compute_logp/consistency"
STALE_FRACTION = "version_stats/stale_token_fraction"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def lookup(stats: Mapping[str, Any], suffix: str) -> float | None:
    """Find a metric by suffix; AReaL may prefix scopes differently per role."""

    if suffix in stats:
        return _number(stats[suffix])
    matches = sorted(key for key in stats if key.endswith("/" + suffix) or key == suffix)
    return _number(stats[matches[0]]) if matches else None


@dataclass
class StepRecord:
    global_step: int
    n_valid: float | None = None
    abs_diff_max: float | None = None
    abs_diff_avg: float | None = None
    mismatch: float | None = None
    nonfinite: float | None = None
    n_same: float | None = None
    same_mismatch: float | None = None
    same_abs_diff_max: float | None = None
    n_stale: float | None = None
    stale_abs_diff_avg: float | None = None
    n_mixed_same: float | None = None
    mixed_same_mismatch: float | None = None
    stale_token_fraction: float | None = None

    @classmethod
    def from_stats(cls, global_step: int, stats: Mapping[str, Any]) -> StepRecord:
        n_valid = lookup(stats, f"{OURS}/n_valid_tokens")
        if n_valid is None:
            n_valid = lookup(stats, "compute_logp/r3/n_valid_tokens")
        return cls(
            global_step=global_step,
            n_valid=n_valid,
            abs_diff_max=lookup(stats, f"{AREAL_ABS_DIFF}/max"),
            abs_diff_avg=lookup(stats, f"{AREAL_ABS_DIFF}/avg"),
            mismatch=lookup(stats, f"{OURS}/mismatch_tokens"),
            nonfinite=lookup(stats, f"{OURS}/nonfinite_tokens"),
            n_same=lookup(stats, f"{OURS}/n_same_version_tokens"),
            same_mismatch=lookup(stats, f"{OURS}/same_version_mismatch_tokens"),
            same_abs_diff_max=lookup(stats, f"{OURS}/same_version_logp_abs_diff/max"),
            n_stale=lookup(stats, f"{OURS}/n_stale_tokens"),
            stale_abs_diff_avg=lookup(stats, f"{OURS}/stale_logp_abs_diff/avg"),
            n_mixed_same=lookup(stats, f"{OURS}/n_mixed_seq_same_version_tokens"),
            mixed_same_mismatch=lookup(stats, f"{OURS}/mixed_seq_same_version_mismatch_tokens"),
            stale_token_fraction=lookup(stats, STALE_FRACTION),
        )


def load_steps(path: str | Path) -> list[StepRecord]:
    """One record per ``global_step``; later commits for a step overwrite earlier keys."""

    merged: dict[int, dict[str, Any]] = {}
    file = Path(path)
    if not file.exists():
        return []
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row.get("global_step", row.get("step", 0)))
        merged.setdefault(step, {}).update(row.get("stats", {}))
    return [StepRecord.from_stats(step, merged[step]) for step in sorted(merged)]


@dataclass
class Provenance:
    files: list[str] = field(default_factory=list)
    calls: dict[str, int] = field(default_factory=dict)
    fallbacks: list[Any] = field(default_factory=list)
    rope_roles: list[str] = field(default_factory=list)

    def missing(self) -> list[str]:
        required = [
            f"{framework}.{module}"
            for framework in ("megatron", "vllm")
            for module in ("attention", "ffn", "logp")
        ]
        absent = [name for name in required if self.calls.get(name, 0) <= 0]
        for role in ("training", "rollout"):
            if role not in self.rope_roles:
                absent.append(f"rope.{role}")
        return absent


def load_provenance(directory: str | Path) -> Provenance:
    result = Provenance()
    root = Path(directory)
    if not root.is_dir():
        return result
    for path in sorted(root.glob("*.json")):
        result.files.append(path.name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("module") == "rope":
            result.rope_roles.append(str(payload.get("role")))
            continue
        framework = str(payload.get("framework", "unknown"))
        result.fallbacks.extend(payload.get("fallbacks", []))
        for module, operator in (payload.get("operators") or {}).items():
            key = f"{framework}.{module}"
            result.calls[key] = result.calls.get(key, 0) + int(operator.get("call_count", 0))
    return result


def _sum(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


@dataclass
class ArmVerdict:
    arm: str
    mode: str
    verdict: str
    reasons: list[str]
    steps: int
    totals: dict[str, float | None]


def _finite_zero(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value == 0.0


def judge_arm(
    arm: str,
    mode: str,
    steps: list[StepRecord],
    *,
    expected_steps: int,
    provenance: Provenance | None,
    min_same_version_tokens: int = 10_000,
) -> ArmVerdict:
    reasons: list[str] = []
    totals = {
        "n_valid": _sum(s.n_valid for s in steps),
        "mismatch": _sum(s.mismatch for s in steps),
        "n_same": _sum(s.n_same for s in steps),
        "same_mismatch": _sum(s.same_mismatch for s in steps),
        "n_mixed_same": _sum(s.n_mixed_same for s in steps),
        "mixed_same_mismatch": _sum(s.mixed_same_mismatch for s in steps),
        "nonfinite": _sum(s.nonfinite for s in steps),
        "max_abs_diff": max((s.abs_diff_max for s in steps if s.abs_diff_max is not None),
                            default=None),
    }
    if arm != "rlkernel":
        if provenance is not None and provenance.files:
            reasons.append(f"native arm unexpectedly wrote RL-Kernel readbacks: {provenance.files}")
            return ArmVerdict(arm, mode, FAIL, reasons, len(steps), totals)
        if totals["max_abs_diff"] == 0.0:
            reasons.append("native arm shows no train/rollout difference; the control is weak")
        return ArmVerdict(arm, mode, REPORT_ONLY, reasons, len(steps), totals)

    if len(steps) < expected_steps:
        reasons.append(f"only {len(steps)} of {expected_steps} steps were committed")
    if provenance is None or provenance.missing():
        missing = ["readback directory"] if provenance is None else provenance.missing()
        reasons.append("strict route not proven to run: " + ", ".join(missing))
    if provenance is not None and provenance.fallbacks:
        reasons.append(f"strict route reported fallbacks: {provenance.fallbacks}")
    if totals["nonfinite"]:
        reasons.append(f"{totals['nonfinite']:.0f} non-finite logp differences")

    if mode == "sync":
        for step in steps:
            if not step.n_valid:
                reasons.append(f"step {step.global_step}: no compared tokens")
            elif step.mismatch is not None:
                if step.mismatch != 0:
                    reasons.append(
                        f"step {step.global_step}: {step.mismatch:.0f} mismatching tokens"
                    )
            elif not _finite_zero(step.abs_diff_max):
                reasons.append(f"step {step.global_step}: max |diff| = {step.abs_diff_max}")
        return ArmVerdict(arm, mode, FAIL if reasons else PASS, reasons, len(steps), totals)

    for step in steps:
        if step.n_same and step.same_mismatch:
            reasons.append(
                f"step {step.global_step}: {step.same_mismatch:.0f} of {step.n_same:.0f} "
                "same-version tokens mismatch"
            )
    if reasons:
        return ArmVerdict(arm, mode, FAIL, reasons, len(steps), totals)
    if (totals["n_same"] or 0) < min_same_version_tokens:
        reasons.append(
            f"only {totals['n_same'] or 0:.0f} same-version tokens "
            f"(< {min_same_version_tokens}); increase steps or lower max_head_offpolicyness"
        )
        return ArmVerdict(arm, mode, INCONCLUSIVE, reasons, len(steps), totals)
    return ArmVerdict(arm, mode, PASS, reasons, len(steps), totals)


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value)}"
    return f"{value:.{digits}g}"


def render_report(
    mode: str,
    arms: Mapping[str, tuple[list[StepRecord], ArmVerdict]],
) -> str:
    lines = [f"# AReaL train/rollout consistency ({mode})", ""]
    lines += ["| arm | verdict | steps | compared tokens | mismatching | max abs diff |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, (_, verdict) in arms.items():
        totals = verdict.totals
        lines.append(
            f"| {name} | {verdict.verdict} | {verdict.steps} | {_fmt(totals['n_valid'])} | "
            f"{_fmt(totals['mismatch'])} | {_fmt(totals['max_abs_diff'])} |"
        )
    if mode == "async":
        lines += ["", "| arm | same-version tokens | same-version mismatching | "
                  "mixed-seq same-version tokens | mixed mismatching |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name, (_, verdict) in arms.items():
            totals = verdict.totals
            lines.append(
                f"| {name} | {_fmt(totals['n_same'])} | {_fmt(totals['same_mismatch'])} | "
                f"{_fmt(totals['n_mixed_same'])} | {_fmt(totals['mixed_same_mismatch'])} |"
            )
    for name, (steps, verdict) in arms.items():
        lines += ["", f"## {name}", ""]
        lines += [f"- {reason}" for reason in verdict.reasons] or ["- no issues"]
        lines += ["", "| step | compared | mismatching | max abs diff | avg abs diff | "
                  "same-version | same mismatching | stale fraction |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for step in steps:
            lines.append(
                f"| {step.global_step} | {_fmt(step.n_valid)} | {_fmt(step.mismatch)} | "
                f"{_fmt(step.abs_diff_max)} | {_fmt(step.abs_diff_avg)} | {_fmt(step.n_same)} | "
                f"{_fmt(step.same_mismatch)} | {_fmt(step.stale_token_fraction)} |"
            )
    return "\n".join(lines) + "\n"


def verdict_payload(arms: Mapping[str, tuple[list[StepRecord], ArmVerdict]]) -> dict[str, Any]:
    return {
        name: {"verdict": asdict(verdict), "steps": [asdict(step) for step in steps]}
        for name, (steps, verdict) in arms.items()
    }


__all__ = [
    "FAIL",
    "INCONCLUSIVE",
    "PASS",
    "REPORT_ONLY",
    "ArmVerdict",
    "Provenance",
    "StepRecord",
    "judge_arm",
    "load_provenance",
    "load_steps",
    "lookup",
    "render_report",
    "verdict_payload",
]
