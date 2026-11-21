# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Performance summary for the AReaL verification arms.

Reads what a run already leaves behind, so it also works on copied results:

* ``stats.jsonl`` (written by ``run_trainer``): AReaL ``timeperf/*`` per step,
  generated tokens, and the wall-clock time of each commit;
* ``spec.json`` / ``run.json``: workload, vLLM execution mode and arm runtime;
* the vLLM ``rollout.log`` under the arm's ``fileroot``: whether CUDA graphs
  were really captured, and vLLM's own generation-throughput samples.

Step 0 includes warm-up (first weight sync, graph capture for the first sizes,
allocator growth), so steady-state means skip the first ``skip`` steps.
Pure Python; no AReaL or torch import.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TIMERS = ("rollout", "train_step", "recompute_logp", "update_weights", "compute_advantage")
GENERATED_TOKENS = ("ppo_actor/n_valid_tokens", "ppo_actor/update/n_valid_tokens")

_THROUGHPUT = re.compile(
    r"Avg generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs"
)
_CAPTURE = re.compile(r"Graph capturing finished in ([0-9.]+) secs?(?:, took ([0-9.]+) GiB)?")
_CUDAGRAPH_MODE = re.compile(r"cudagraph_mode['\"]?\s*[:=]\s*<?(?:CUDAGraphMode\.)?([A-Z_]+)")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


@dataclass
class PerfStep:
    global_step: int
    timers: dict[str, float | None]
    generated_tokens: float | None
    wall_time: float | None
    trainer_started: float | None
    step_seconds: float | None = None  # wall time since the previous commit


def load_perf_steps(path: str | Path) -> list[PerfStep]:
    file = Path(path)
    if not file.exists():
        return []
    merged: dict[int, dict[str, Any]] = {}
    wall: dict[int, float] = {}
    started: float | None = None
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row.get("global_step", row.get("step", 0)))
        merged.setdefault(step, {}).update(row.get("stats", {}))
        stamp = _number(row.get("wall_time"))
        if stamp is not None:
            wall[step] = max(stamp, wall.get(step, stamp))
        if started is None:
            started = _number(row.get("trainer_started"))
    steps: list[PerfStep] = []
    previous: float | None = None
    for step in sorted(merged):
        stats = merged[step]
        tokens = None
        for key in GENERATED_TOKENS:
            tokens = _number(stats.get(key))
            if tokens is not None:
                break
        record = PerfStep(
            global_step=step,
            timers={name: _number(stats.get(f"timeperf/{name}")) for name in TIMERS},
            generated_tokens=tokens,
            wall_time=wall.get(step),
            trainer_started=started,
        )
        if record.wall_time is not None and previous is not None:
            record.step_seconds = record.wall_time - previous
        previous = record.wall_time if record.wall_time is not None else previous
        steps.append(record)
    return steps


@dataclass
class VllmLogEvidence:
    log: str | None = None
    eager: bool | None = None  # True when vLLM logged that CUDA graphs are disabled
    cudagraph_mode: str | None = None
    capture_seconds: float | None = None
    capture_gib: float | None = None
    gen_throughput_samples: int = 0
    gen_throughput_median: float | None = None  # tokens/s, samples with running requests
    gen_throughput_p90: float | None = None


def find_rollout_log(arm_dir: str | Path, trial_name: str | None = None) -> Path | None:
    logs = sorted(Path(arm_dir).glob("fileroot/logs/**/rollout.log"))
    if trial_name:
        named = [path for path in logs if path.parent.name == trial_name]
        if named:
            return named[-1]
    return max(logs, key=lambda path: path.stat().st_mtime) if logs else None


def parse_vllm_log(path: str | Path | None) -> VllmLogEvidence:
    evidence = VllmLogEvidence()
    if path is None or not Path(path).is_file():
        return evidence
    evidence.log = str(path)
    samples: list[float] = []
    eager_seen = False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "Cudagraph is disabled" in line or "Enforce eager set" in line:
            eager_seen = True
        if evidence.cudagraph_mode is None:
            match = _CUDAGRAPH_MODE.search(line)
            if match:
                evidence.cudagraph_mode = match.group(1)
        match = _CAPTURE.search(line)
        if match:
            evidence.capture_seconds = (evidence.capture_seconds or 0.0) + float(match.group(1))
            if match.group(2):
                evidence.capture_gib = (evidence.capture_gib or 0.0) + float(match.group(2))
        match = _THROUGHPUT.search(line)
        if match and int(match.group(2)) > 0 and float(match.group(1)) > 0:
            samples.append(float(match.group(1)))
    evidence.eager = eager_seen
    if samples:
        ordered = sorted(samples)
        evidence.gen_throughput_samples = len(ordered)
        evidence.gen_throughput_median = statistics.median(ordered)
        evidence.gen_throughput_p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    return evidence


@dataclass
class ArmPerf:
    label: str
    arm: str
    mode: str
    vllm_execution: str
    rollout_gpus: int
    steps: int
    skip: int
    elapsed_seconds: float | None
    startup_seconds: float | None  # trainer start -> first committed step
    means: dict[str, float | None]  # steady-state means
    per_step: list[dict[str, Any]] = field(default_factory=list)
    vllm: VllmLogEvidence = field(default_factory=VllmLogEvidence)
    warnings: list[str] = field(default_factory=list)


def _execution(spec: Mapping[str, Any]) -> str:
    value = spec.get("vllm_execution")
    if isinstance(value, str):
        return value
    eager = spec.get("vllm_enforce_eager")  # results from bundles before vllm_execution
    return "eager" if eager in (None, True) else "graph"


def summarize_arm(arm_dir: str | Path, *, label: str | None = None, skip: int = 1) -> ArmPerf:
    arm_dir = Path(arm_dir)
    run = {}
    if (arm_dir / "run.json").is_file():
        run = json.loads((arm_dir / "run.json").read_text(encoding="utf-8"))
    spec = run.get("spec") or {}
    if not spec and (arm_dir.parent / "spec.json").is_file():
        spec = json.loads((arm_dir.parent / "spec.json").read_text(encoding="utf-8"))
    steps = load_perf_steps(arm_dir / "stats.jsonl")
    steady = steps[skip:] if len(steps) > skip else steps
    rollout_gpus = int(spec.get("rollout_tp", 1)) * int(spec.get("rollout_dp", 1))
    execution = _execution(spec)
    mode = str(spec.get("mode", "?"))
    warnings: list[str] = []
    if mode == "sync" and steps and all(step.wall_time is None for step in steps):
        # Results from bundles before run_trainer recorded wall time: in sync mode the
        # phases run back to back, so their sum approximates the step time.
        for step in steps:
            parts = [value for value in step.timers.values() if value is not None]
            step.step_seconds = sum(parts) if parts else None
        warnings.append("no wall-clock stamps; step time = sum of timeperf phases")
    means: dict[str, float | None] = {
        name: _mean(step.timers.get(name) for step in steady) for name in TIMERS
    }
    means["step_seconds"] = _mean(step.step_seconds for step in steady)
    means["generated_tokens"] = _mean(step.generated_tokens for step in steady)
    # In async mode timeperf/rollout is only the wait for a ready batch.
    rollout_tps = [] if mode == "async" else [
        step.generated_tokens / step.timers["rollout"]
        for step in steady
        if step.generated_tokens and step.timers.get("rollout")
    ]
    means["rollout_tokens_per_s"] = _mean(rollout_tps)
    means["rollout_tokens_per_gpu_s"] = (
        means["rollout_tokens_per_s"] / rollout_gpus
        if means["rollout_tokens_per_s"] is not None and rollout_gpus
        else None
    )
    startup = None
    if steps and steps[0].wall_time is not None and steps[0].trainer_started is not None:
        startup = steps[0].wall_time - steps[0].trainer_started
    perf = ArmPerf(
        label=label or arm_dir.name,
        arm=str(run.get("arm", arm_dir.name)),
        mode=mode,
        vllm_execution=execution,
        rollout_gpus=rollout_gpus,
        steps=len(steps),
        skip=min(skip, len(steps)),
        elapsed_seconds=_number(run.get("elapsed_seconds")),
        startup_seconds=startup,
        means=means,
        per_step=[
            {
                "global_step": step.global_step,
                "step_seconds": step.step_seconds,
                "generated_tokens": step.generated_tokens,
                **{f"timeperf/{name}": step.timers.get(name) for name in TIMERS},
            }
            for step in steps
        ],
        vllm=parse_vllm_log(find_rollout_log(arm_dir, run.get("trial_name"))),
        warnings=warnings,
    )
    if execution == "graph" and perf.vllm.eager:
        perf.warnings.append("graph requested but vLLM logged that CUDA graphs are disabled")
    if execution == "eager" and perf.vllm.eager is False and perf.vllm.log:
        perf.warnings.append("eager requested but vLLM did not log eager mode")
    if perf.vllm.log is None:
        perf.warnings.append("rollout.log not found; graph evidence unavailable")
    return perf


def paired_ratio_ci(
    baseline: Sequence[float | None],
    candidate: Sequence[float | None],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    """Ratio of means candidate/baseline with a paired bootstrap 95% interval.

    Steps are paired by index (same seed and data order in both arms).
    """

    count = min(len(baseline), len(candidate))  # arms may commit different step counts
    pairs = [
        (b, c)
        for b, c in zip(baseline[:count], candidate[:count], strict=True)
        if b is not None and c is not None
    ]
    if not pairs or sum(b for b, _ in pairs) == 0:
        return None, None, None
    ratio = sum(c for _, c in pairs) / sum(b for b, _ in pairs)
    if len(pairs) < 3:
        return ratio, None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        base = sum(b for b, _ in sample)
        if base:
            draws.append(sum(c for _, c in sample) / base)
    draws.sort()
    return ratio, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def _steady(perf: ArmPerf, key: str) -> list[float | None]:
    return [row.get(key) for row in perf.per_step[perf.skip:]]


# (label, key in means, per-step key or None, higher is better, unit)
METRICS: tuple[tuple[str, str, str | None, bool, str], ...] = (
    ("Rollout time", "rollout", "timeperf/rollout", False, "s/step"),
    ("Rollout throughput", "rollout_tokens_per_gpu_s", None, True, "tok/GPU/s"),
    ("vLLM log generation throughput, coarse median", "vllm_gen_median", None, True, "tok/s"),
    ("Recompute logp time", "recompute_logp", "timeperf/recompute_logp", False, "s/step"),
    ("Actor train time", "train_step", "timeperf/train_step", False, "s/step"),
    ("Weight update time", "update_weights", "timeperf/update_weights", False, "s/step"),
    ("End-to-end step time", "step_seconds", "step_seconds", False, "s/step"),
)


ASYNC_MEANINGLESS = frozenset({"rollout", "rollout_tokens_per_gpu_s"})


def _value(perf: ArmPerf, key: str) -> float | None:
    if key == "vllm_gen_median":
        return perf.vllm.gen_throughput_median
    return perf.means.get(key)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _relative(base: float | None, cand: float | None, higher_is_better: bool) -> str:
    if base is None or cand is None or base == 0:
        return "-"
    ratio = cand / base
    if higher_is_better:
        return f"{ratio:.2f}x ({(ratio - 1) * 100:+.1f}%)"
    word = "slower" if ratio > 1 else "faster"
    return f"{ratio:.2f}x ({abs(ratio - 1) * 100:.1f}% {word})"


def comparison_table(baseline: ArmPerf, candidate: ArmPerf) -> list[str]:
    lines = [
        f"| Metric | {baseline.label} | {candidate.label} | {candidate.label} / "
        f"{baseline.label} | paired 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key, step_key, higher, unit in METRICS:
        base, cand = _value(baseline, key), _value(candidate, key)
        ci = "-"
        if key in ASYNC_MEANINGLESS and "async" in (baseline.mode, candidate.mode):
            base = cand = None  # async rollout overlaps training; timer is only the wait
        elif step_key is not None:
            _, low, high = paired_ratio_ci(
                _steady(baseline, step_key), _steady(candidate, step_key)
            )
            if low is not None:
                ci = f"[{low:.2f}x, {high:.2f}x]"
        lines.append(
            f"| {label} ({unit}) | {_fmt(base)} | {_fmt(cand)} | "
            f"{_relative(base, cand, higher)} | {ci} |"
        )
    return lines


def arm_table(arms: Sequence[ArmPerf]) -> list[str]:
    lines = [
        "| arm | mode | vLLM | graph evidence | steps (skip) | startup s | elapsed s | "
        "rollout s | train s | recompute s | step s | tok/GPU/s | vLLM gen tok/s |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for perf in arms:
        if perf.vllm.log is None:
            evidence = "no log"
        elif perf.vllm.eager:
            evidence = "eager (graphs disabled)"
        else:
            evidence = perf.vllm.cudagraph_mode or "graphs on"
            if perf.vllm.capture_seconds is not None:
                evidence += f", capture {perf.vllm.capture_seconds:.0f}s"
        means = perf.means
        rollout = "-" if perf.mode == "async" else _fmt(means.get("rollout"))
        lines.append(
            f"| {perf.label} | {perf.mode} | {perf.vllm_execution} | {evidence} | "
            f"{perf.steps} ({perf.skip}) | {_fmt(perf.startup_seconds, 0)} | "
            f"{_fmt(perf.elapsed_seconds, 0)} | {rollout} | "
            f"{_fmt(means.get('train_step'))} | {_fmt(means.get('recompute_logp'))} | "
            f"{_fmt(means.get('step_seconds'))} | "
            f"{_fmt(means.get('rollout_tokens_per_gpu_s'), 1)} | "
            f"{_fmt(perf.vllm.gen_throughput_median, 0)} |"
        )
    return lines


def perf_payload(arms: Sequence[ArmPerf]) -> dict[str, Any]:
    return {perf.label: asdict(perf) for perf in arms}


__all__ = [
    "ArmPerf",
    "METRICS",
    "PerfStep",
    "VllmLogEvidence",
    "arm_table",
    "comparison_table",
    "find_rollout_log",
    "load_perf_steps",
    "paired_ratio_ci",
    "parse_vllm_log",
    "perf_payload",
    "summarize_arm",
]
