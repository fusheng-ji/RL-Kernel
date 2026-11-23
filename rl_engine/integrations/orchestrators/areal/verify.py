# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Verify AReaL train/rollout log-probability consistency with RL-Kernel.

Subcommands::

    inventory  record packages, GPUs and source revisions (no GPU work)
    preflight  check every AReaL / Megatron / vLLM symbol the adapter patches
    run        run AReaL's GSM8K example for the native and RL-Kernel arms
    analyze    rebuild the verdict and report from an existing output directory
    perf       collect every stage under a run root into one performance report

The judgement uses AReaL's own ``compute_logp/r3/rollout_train_logp_abs_diff``
metric plus the adapter's exact mismatch counts. AReaL source is not modified;
the RL-Kernel arm only changes the actor worker command and environment.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rl_engine.integrations.orchestrators.areal import analysis, launch, perf

PACKAGES = (
    "torch",
    "vllm",
    "megatron-core",
    "transformer-engine",
    "mbridge",
    "flash-attn",
    "transformers",
    "areal",
    "rl-kernel",
    "rl_engine",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _git_revision(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"path": str(path), "head": run("rev-parse", "HEAD"), "dirty": run("status", "--short")}


# --------------------------------------------------------------------------- inventory


def _ray_status() -> str | None:
    """``ray status`` of the cluster named by ``RAY_ADDRESS``, or None without one."""

    address = os.environ.get("RAY_ADDRESS", "").strip()
    if not address:
        return None
    try:
        return subprocess.check_output(
            ["ray", "status", "--address", address],
            text=True, stderr=subprocess.STDOUT, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def cmd_inventory(args: argparse.Namespace) -> int:
    out = Path(args.out)
    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "rl_kernel": _git_revision(Path(__file__).resolve().parents[4]),
    }
    if args.areal_root:
        payload["areal"] = _git_revision(Path(args.areal_root))
    try:
        payload["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total",
             "--format=csv"],
            text=True,
        )
        payload["gpu_processes"] = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        payload["nvidia_smi"] = f"unavailable: {error}"
    payload["ray_status"] = _ray_status()
    _write_json(out / "inventory.json", payload)
    print(json.dumps(payload["packages"], indent=2))
    return 0


# --------------------------------------------------------------------------- preflight


def _params(func: Any) -> list[str]:
    return list(inspect.signature(func).parameters)


def _require_params(func: Any, names: list[str]) -> str:
    params = _params(func)
    missing = [name for name in names if name not in params]
    if missing:
        raise AssertionError(f"missing parameters {missing}; signature has {params}")
    return f"params={params}"


def preflight_checks(areal_root: Path | None, model: Path | None) -> list[tuple[str, Callable]]:
    checks: list[tuple[str, Callable[[], str]]] = []

    def add(name: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
        def register(func: Callable[[], str]) -> Callable[[], str]:
            checks.append((name, func))
            return func

        return register

    @add("areal.megatron_engine patch targets")
    def _engine() -> str:
        module = importlib.import_module("areal.engine.megatron_engine")
        for name in ("make_hf_and_mcore_config", "make_mcore_model", "unwrap_to_gpt_model",
                     "ChunkedLMHeadOutput", "chunked_lm_head_logprobs_entropy"):
            if not hasattr(module, name):
                raise AssertionError(f"areal.engine.megatron_engine.{name} is missing")
        fields = list(getattr(module.ChunkedLMHeadOutput, "_fields", ()))
        expected = ["logprobs", "entropy", "vocab_min_logits", "vocab_max_logits",
                    "vocab_mean_logits", "vocab_norm_logits"]
        if fields != expected:
            raise AssertionError(f"ChunkedLMHeadOutput fields changed: {fields}")
        _require_params(module.MegatronEngine.initialize, ["self"])
        return _require_params(
            module.chunked_lm_head_logprobs_entropy,
            ["output_layer", "hidden_states", "weight", "labels", "temperature",
             "chunk_size", "logit_scale"],
        )

    @add("areal actor consistency metric")
    def _actor() -> str:
        module = importlib.import_module("areal.trainer.ppo.actor")
        return _require_params(
            module.PPOActor._log_r3_rollout_train_logp_stats,
            ["train_logp", "data", "r3_enabled"],
        )

    @add("areal stats logger / rpc server")
    def _logger() -> str:
        module = importlib.import_module("areal.utils.stats_logger")
        _require_params(module.StatsLogger.commit, ["epoch", "step", "global_step", "data"])
        if importlib.util.find_spec("areal.infra.rpc.rpc_server") is None:
            raise AssertionError("areal.infra.rpc.rpc_server is not importable")
        return "ok"

    @add("megatron rotary / attention / mlp")
    def _megatron() -> str:
        attention = importlib.import_module("megatron.core.transformer.attention")
        rope_params = _params(attention.apply_rotary_pos_emb)
        if len(rope_params) < 3:
            raise AssertionError(f"apply_rotary_pos_emb signature changed: {rope_params}")
        detail = f"apply_rotary_pos_emb{rope_params}"
        rotary = importlib.import_module(
            "megatron.core.models.common.embeddings.rotary_pos_embedding"
        )
        _require_params(rotary.RotaryEmbedding.__init__, ["rotary_base", "rotary_percent"])
        from rl_engine.integrations.engines.train.megatron.runtime import (
            _discover_attention_classes,
            _discover_ffn_classes,
        )

        attention_classes = [cls.__name__ for cls in _discover_attention_classes()]
        ffn_classes = [cls.__name__ for cls in _discover_ffn_classes()]
        if not attention_classes or not ffn_classes:
            raise AssertionError(f"attention={attention_classes} ffn={ffn_classes}")
        return f"{detail}; attention={attention_classes}; ffn={ffn_classes}"

    @add("vllm sampler / rotary / qwen3 / plugin")
    def _vllm() -> str:
        sampler = importlib.import_module("vllm.v1.sample.sampler")
        if not hasattr(sampler.Sampler, "forward"):
            raise AssertionError("vllm.v1.sample.sampler.Sampler.forward is missing")
        rotary = importlib.import_module("vllm.model_executor.layers.rotary_embedding")
        detail = _require_params(rotary.RotaryEmbedding.forward_cuda, ["positions", "query"])
        importlib.import_module("vllm.model_executor.models.qwen3").Qwen3Attention  # noqa: B018
        plugins = [
            entry.value
            for entry in importlib.metadata.entry_points(group="vllm.general_plugins")
            if entry.name == "rl_kernel"
        ]
        if not plugins:
            raise AssertionError("rl_kernel vllm.general_plugins entry point is not installed")
        return f"vllm={importlib.metadata.version('vllm')}; forward_cuda {detail}; {plugins}"

    @add("vllm CUDA graph configuration")
    def _vllm_graph() -> str:
        compilation = importlib.import_module("vllm.config.compilation")
        mode = compilation.CUDAGraphMode.FULL_DECODE_ONLY
        fields = getattr(compilation.CompilationConfig, "__dataclass_fields__", {})
        missing = [name for name in ("cudagraph_mode", "cudagraph_capture_sizes",
                                     "max_cudagraph_capture_size") if name not in fields]
        engine_args = importlib.import_module("vllm.engine.arg_utils").EngineArgs
        if "optimization_level" not in getattr(engine_args, "__dataclass_fields__", {}):
            missing.append("EngineArgs.optimization_level")
        if missing:
            raise AssertionError(f"missing vLLM config fields: {missing}")
        return f"{mode}; capture-size fields present"

    @add("areal vLLM server argument builder")
    def _areal_vllm_args() -> str:
        module = importlib.import_module("areal.api.cli_args")
        _require_params(module.vLLMConfig.build_args, ["vllm_config", "tp_size", "pp_size"])
        if not hasattr(module.vLLMConfig, "build_cmd_from_args"):
            raise AssertionError("vLLMConfig.build_cmd_from_args is missing")
        return "ok"

    @add("FlashAttention-4 (strict Attention)")
    def _fa4() -> str:
        errors = []
        for source in ("flash_attn.cute.interface", "vllm.vllm_flash_attn.cute.interface"):
            try:
                importlib.import_module(source)
                return f"importable: {source}"
            except Exception as error:  # noqa: BLE001 - CuTe import can raise anything
                errors.append(f"{source}: {type(error).__name__}: {error}")
        raise AssertionError("; ".join(errors))

    @add("rl_engine CUDA extension")
    def _extension() -> str:
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE

        missing = [name for name in ("rope_apply_sm90",) if not hasattr(_C, name)]
        if not _EXT_AVAILABLE or missing:
            raise AssertionError(f"extension available={_EXT_AVAILABLE} missing={missing}")
        return "ok"

    if model is not None:

        @add("model checkpoint")
        def _model() -> str:
            config = json.loads((model / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != "qwen3":
                raise AssertionError(f"model_type={config.get('model_type')}")
            if config.get("tie_word_embeddings"):
                raise AssertionError("tied embeddings are unsupported")
            return f"vocab_size={config.get('vocab_size')}"

    if areal_root is not None:

        @add("areal example files")
        def _example() -> str:
            for relative in (launch.DEFAULT_EXAMPLE, launch.DEFAULT_CONFIG):
                if not (areal_root / relative).is_file():
                    raise AssertionError(f"{relative} not found under {areal_root}")
            return "ok"

    return checks


def cmd_preflight(args: argparse.Namespace) -> int:
    areal_root = Path(args.areal_root) if args.areal_root else None
    model = Path(args.model) if args.model else None
    results = []
    for name, check in preflight_checks(areal_root, model):
        try:
            detail = check()
            results.append({"check": name, "ok": True, "detail": detail})
        except Exception as error:  # noqa: BLE001 - report every failing check
            detail = f"{type(error).__name__}: {error}"
            results.append({"check": name, "ok": False, "detail": detail})
    for result in results:
        print(f"[{'OK' if result['ok'] else 'FAIL'}] {result['check']}: {result['detail']}")
    if args.out:
        _write_json(Path(args.out) / "preflight.json", results)
    return 0 if all(result["ok"] for result in results) else 1


# --------------------------------------------------------------------------- run / analyze


def _spec_from_args(args: argparse.Namespace) -> launch.ExperimentSpec:
    vocab, _tied = launch.read_vocabulary(args.model)
    extra_env = dict(item.split("=", 1) for item in args.env or [])
    return launch.ExperimentSpec(
        model=str(Path(args.model).resolve()),
        mode=args.mode,
        steps=args.steps,
        seed=args.seed,
        train_tp=args.train_tp,
        train_cp=args.train_cp,
        rollout_tp=args.rollout_tp,
        rollout_dp=args.rollout_dp,
        n_gpus_per_node=args.gpus,
        n_nodes=args.nodes,
        scheduler=args.scheduler,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        max_offpolicyness=args.max_offpolicyness,
        max_num_seqs=args.max_num_seqs,
        vllm_execution=args.vllm_execution,
        real_vocab_size=args.real_vocab_size or vocab,
        padded_vocab_size=args.padded_vocab_size or vocab,
        dataset_path=args.dataset_path,
        extra_env=extra_env,
        profile_steps=tuple(
            sorted({int(item) for item in (args.profile_steps or "").split(",") if item})
        ),
        native_observe=args.native_observe,
        native_rope_fusion=args.native_rope_fusion,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_arm(spec: launch.ExperimentSpec, arm: str, *, areal_root: Path, out: Path,
            base_config: Path, timeout_minutes: float, dry_run: bool) -> dict[str, Any]:
    arm_dir = (out / arm).resolve()
    arm_dir.mkdir(parents=True, exist_ok=True)
    readback = arm_dir / "readback"
    stats = arm_dir / "stats.jsonl"
    if stats.exists():
        stats.unlink()
    arm_env = launch.arm_environment(spec, arm, readback_dir=readback, stats_path=stats)
    trial = f"{arm}-{time.strftime('%Y%m%d-%H%M%S')}"
    config = launch.build_config(
        _load_yaml(base_config), spec, arm, arm_dir=arm_dir, env_vars=arm_env, trial_name=trial
    )
    config_path = arm_dir / "config.yaml"
    _dump_yaml(config_path, config)
    command = [
        sys.executable,
        "-m",
        "rl_engine.integrations.orchestrators.areal.run_trainer",
        str(areal_root / launch.DEFAULT_EXAMPLE),
        "--config",
        str(config_path),
    ]
    env = launch.scrubbed_environment(os.environ)
    env.update(arm_env)
    record: dict[str, Any] = {
        "arm": arm,
        "command": command,
        "cwd": str(areal_root),
        "arm_env": arm_env,
        "spec": asdict(spec),
        "trial_name": trial,
    }
    print(f"[verify] arm={arm} config={config_path}", flush=True)
    if dry_run:
        record["returncode"] = None
        _write_json(arm_dir / "run.json", record)
        return record
    record["started"] = time.time()
    with (arm_dir / "trainer.log").open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=str(areal_root),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_minutes * 60 if timeout_minutes > 0 else None,
                check=False,
            )
            record["returncode"] = completed.returncode
        except subprocess.TimeoutExpired:
            record["returncode"] = "timeout"
    record["finished"] = time.time()
    record["elapsed_seconds"] = record["finished"] - record["started"]
    _write_json(arm_dir / "run.json", record)
    print(f"[verify] arm={arm} returncode={record['returncode']} "
          f"elapsed={record['elapsed_seconds']:.0f}s", flush=True)
    return record


def analyze(out: Path, mode: str, arms: list[str], expected_steps: int,
            min_same_version_tokens: int) -> int:
    results: dict[str, tuple[list[analysis.StepRecord], analysis.ArmVerdict]] = {}
    for arm in arms:
        arm_dir = out / arm
        steps = analysis.load_steps(arm_dir / "stats.jsonl")
        readback = arm_dir / "readback"
        provenance = analysis.load_provenance(readback) if readback.exists() else None
        if arm != launch.ARM_RLKERNEL and provenance is None:
            provenance = analysis.Provenance()
        verdict = analysis.judge_arm(
            arm, mode, steps, expected_steps=expected_steps, provenance=provenance,
            min_same_version_tokens=min_same_version_tokens,
        )
        run_file = arm_dir / "run.json"
        if run_file.exists():
            run = json.loads(run_file.read_text(encoding="utf-8"))
            if run.get("returncode") not in (0, None):
                verdict.reasons.insert(0, f"trainer exited with {run.get('returncode')}")
                if verdict.verdict == analysis.PASS:
                    verdict.verdict = analysis.FAIL
        results[arm] = (steps, verdict)
    report = analysis.render_report(mode, results)
    arm_perf = [perf.summarize_arm(out / arm, label=arm) for arm in arms if (out / arm).is_dir()]
    if arm_perf:
        report += "\n" + "\n".join(_perf_section(arm_perf)) + "\n"
        _write_json(out / "perf.json", perf.perf_payload(arm_perf))
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    _write_json(out / "summary.json", analysis.verdict_payload(results))
    print(report)
    strict = results.get(launch.ARM_RLKERNEL)
    if strict is None:
        return 0
    return {analysis.PASS: 0, analysis.INCONCLUSIVE: 2}.get(strict[1].verdict, 1)


def _perf_section(arm_perf: list[perf.ArmPerf]) -> list[str]:
    lines = ["## Performance (steady state, first step skipped)", ""]
    lines += perf.arm_table(arm_perf)
    by_arm = {item.arm: item for item in arm_perf}
    if launch.ARM_NATIVE in by_arm and launch.ARM_RLKERNEL in by_arm:
        native, rlkernel = by_arm[launch.ARM_NATIVE], by_arm[launch.ARM_RLKERNEL]
        lines += [""] + perf.comparison_table(native, rlkernel)
    for item in arm_perf:
        lines += [f"- {item.label}: {warning}" for warning in item.warnings]
    return lines


def _stage_arms(root: Path, skip: int) -> list[tuple[str, perf.ArmPerf, dict[str, Any]]]:
    """Every arm under every stage directory of a run root."""

    found = []
    for stage_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (stage_dir / "spec.json").is_file():
            continue
        summary = {}
        if (stage_dir / "summary.json").is_file():
            summary = json.loads((stage_dir / "summary.json").read_text(encoding="utf-8"))
        for arm in launch.ARMS:
            if not (stage_dir / arm / "stats.jsonl").is_file():
                continue
            item = perf.summarize_arm(stage_dir / arm, skip=skip)
            item.label = f"{arm}-{item.vllm_execution}"
            found.append((stage_dir.name, item, summary.get(arm, {})))
    return found


COMPARISONS = (
    ("RL-Kernel cost with CUDA graphs on both arms", "native-graph", "rlkernel-graph"),
    ("CUDA graph gain on the RL-Kernel arm", "rlkernel-eager", "rlkernel-graph"),
    ("CUDA graph gain on the native arm", "native-eager", "native-graph"),
    ("Previous configuration: native eager vs RL-Kernel graph", "native-eager", "rlkernel-graph"),
    ("RL-Kernel cost, both arms eager (previous runs)", "native-eager", "rlkernel-eager"),
)


def render_perf_report(root: Path, skip: int) -> tuple[str, dict[str, Any]]:
    found = _stage_arms(root, skip)
    lines = [f"# AReaL x RL-Kernel performance and consistency: {root.name}", ""]
    if not found:
        return "\n".join(lines + ["No stage with stats.jsonl found.", ""]), {}

    lines += ["## Consistency", "",
              "| stage | arm | verdict | steps | compared tokens | mismatching | "
              "same-version tokens | same-version mismatching | max abs diff |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for stage, item, summary in found:
        verdict = summary.get("verdict", {})
        totals = verdict.get("totals", {})
        sync = item.mode == "sync"
        same = "-" if sync else analysis._fmt(totals.get("n_same"))
        same_mismatch = "-" if sync else analysis._fmt(totals.get("same_mismatch"))
        lines.append(
            f"| {stage} | {item.label} | {verdict.get('verdict', '-')} | {item.steps} | "
            f"{analysis._fmt(totals.get('n_valid'))} | {analysis._fmt(totals.get('mismatch'))} | "
            f"{same} | {same_mismatch} | {analysis._fmt(totals.get('max_abs_diff'))} |"
        )

    lines += ["", f"## Per arm (steady state, first {skip} step(s) skipped)", ""]
    for stage in dict.fromkeys(stage for stage, _, _ in found):
        lines += [f"### {stage}", ""]
        lines += perf.arm_table([item for name, item, _ in found if name == stage]) + [""]

    by_key = {(item.mode, item.label): item for _, item, _ in found}
    lines += ["## Comparisons", ""]
    for mode in ("sync", "async"):
        for title, base, cand in COMPARISONS:
            if (mode, base) in by_key and (mode, cand) in by_key:
                lines += [f"### {mode}: {title}", ""]
                lines += perf.comparison_table(by_key[(mode, base)], by_key[(mode, cand)]) + [""]

    warnings = [f"- {stage}/{item.label}: {warning}"
                for stage, item, _ in found for warning in item.warnings]
    if warnings:
        lines += ["## Warnings", ""] + warnings + [""]
    lines += ["## Notes", "",
              "- Rollout throughput = generated tokens / timeperf/rollout / rollout GPUs.",
              "- vLLM generation throughput = median of vLLM's 10 s log samples with running "
              "requests.",
              "- Paired CI: bootstrap over steps paired by index (same seed and data order).",
              "- Async rollout overlaps training, so compare async arms by end-to-end step time.",
              ""]
    payload = {f"{stage}/{item.label}": asdict(item) for stage, item, _ in found}
    return "\n".join(lines), payload


def cmd_perf(args: argparse.Namespace) -> int:
    root = Path(args.out).resolve()
    report, payload = render_perf_report(root, args.skip)
    (root / "PERF.md").write_text(report, encoding="utf-8")
    _write_json(root / "perf.json", payload)
    print(report)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    areal_root = Path(args.areal_root).resolve()
    out = Path(args.out).resolve()
    spec = _spec_from_args(args)
    spec.validate()
    base_config = Path(args.base_config).resolve() if args.base_config else (
        areal_root / launch.DEFAULT_CONFIG
    )
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    for arm in arms:
        if arm not in launch.ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {launch.ARMS}")
    _write_json(out / "spec.json", asdict(spec))
    for arm in arms:  # sequential: the arms share the same GPUs
        run_arm(spec, arm, areal_root=areal_root, out=out, base_config=base_config,
                timeout_minutes=args.timeout_minutes, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    return analyze(out, spec.mode, arms, spec.steps, args.min_same_version_tokens)


def cmd_analyze(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    spec = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    arms = [arm for arm in launch.ARMS if (out / arm).is_dir()]
    return analyze(out, spec["mode"], arms, int(spec["steps"]), args.min_same_version_tokens)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory")
    inventory.add_argument("--out", required=True)
    inventory.add_argument("--areal-root")
    inventory.set_defaults(func=cmd_inventory)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--areal-root")
    preflight.add_argument("--model")
    preflight.add_argument("--out")
    preflight.set_defaults(func=cmd_preflight)

    run = sub.add_parser("run")
    run.add_argument("--areal-root", required=True)
    run.add_argument("--model", required=True, help="local Qwen3 HF checkpoint directory")
    run.add_argument("--out", required=True)
    run.add_argument("--mode", choices=(launch.MODE_SYNC, launch.MODE_ASYNC), default="sync")
    run.add_argument("--arms", default="native,rlkernel")
    run.add_argument("--steps", type=int, default=5)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--gpus", type=int, default=8, help="GPUs per node")
    run.add_argument("--nodes", type=int, default=1,
                     help="number of nodes; more than one requires --scheduler ray")
    run.add_argument("--scheduler", choices=launch.SCHEDULERS, default=launch.SCHEDULER_LOCAL,
                     help="AReaL scheduler.type")
    run.add_argument("--train-tp", type=int, default=4)
    run.add_argument("--train-cp", type=int, default=1)
    run.add_argument("--rollout-tp", type=int, default=4)
    run.add_argument("--rollout-dp", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--n-samples", type=int, default=4)
    run.add_argument("--max-new-tokens", type=int, default=1024)
    run.add_argument("--max-offpolicyness", type=int, default=1,
                     help="rollout.max_head_offpolicyness in async mode")
    run.add_argument("--max-num-seqs", type=int, default=128)
    run.add_argument("--vllm-execution", choices=launch.VLLM_EXECUTIONS, default=launch.VLLM_EAGER,
                     help="eager: vLLM enforce_eager; graph: -O0 + FULL_DECODE_ONLY CUDA graphs "
                          "captured at every batch size up to --max-num-seqs (both arms)")
    run.add_argument("--real-vocab-size", type=int)
    run.add_argument("--padded-vocab-size", type=int)
    run.add_argument("--env", action="append", help="extra KEY=VALUE for the RL-Kernel arm")
    run.add_argument("--profile-steps", default="",
                     help="comma-separated actor versions to torch.profile on rank 0 (both arms)")
    run.add_argument("--native-observe", action="store_true",
                     help="add the same-version consistency metrics to the native arm in sync "
                          "mode as well (async native arms always observe); numerics unchanged")
    run.add_argument("--native-rope-fusion", action="store_true",
                     help="enable Megatron apply_rope_fusion on the native arm via the RL-Kernel "
                          "worker (TE fused THD RoPE); numerics differ from the unfused baseline")
    run.add_argument("--dataset-path",
                     help="local GSM8K dataset (default: openai/gsm8k from the HF hub)")
    run.add_argument("--base-config", help=f"default: <areal-root>/{launch.DEFAULT_CONFIG}")
    run.add_argument("--timeout-minutes", type=float, default=0)
    run.add_argument("--min-same-version-tokens", type=int, default=10_000)
    run.add_argument("--dry-run", action="store_true", help="write configs only")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("analyze")
    report.add_argument("--out", required=True)
    report.add_argument("--min-same-version-tokens", type=int, default=10_000)
    report.set_defaults(func=cmd_analyze)

    perf_report = sub.add_parser("perf")
    perf_report.add_argument("--out", required=True, help="run root containing the stage dirs")
    perf_report.add_argument("--skip", type=int, default=1, help="warm-up steps to skip")
    perf_report.set_defaults(func=cmd_perf)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
