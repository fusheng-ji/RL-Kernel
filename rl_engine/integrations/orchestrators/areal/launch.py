# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Build AReaL experiment configs and environments for the verification arms.

The verifier edits a copy of AReaL's own example YAML instead of passing Hydra
overrides, because list entries such as ``actor.scheduling_spec[0].cmd`` cannot
be overridden from the command line. Pure Python; no AReaL import.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKER_CMD = "python3 -m rl_engine.integrations.orchestrators.areal.worker"
DEFAULT_RPC_CMD = "python3 -m areal.infra.rpc.rpc_server"
DEFAULT_EXAMPLE = "examples/math/gsm8k_rl.py"
DEFAULT_CONFIG = "examples/math/gsm8k_grpo_megatron.yaml"

ARM_NATIVE = "native"
ARM_RLKERNEL = "rlkernel"
ARMS = (ARM_NATIVE, ARM_RLKERNEL)
MODE_SYNC = "sync"
MODE_ASYNC = "async"

SCHEDULER_LOCAL = "local"
SCHEDULER_RAY = "ray"
SCHEDULERS = (SCHEDULER_LOCAL, SCHEDULER_RAY)

# vLLM execution for the rollout server. "graph" mirrors the VIME CUDA setup of
# RL-Kernel PR 377: no torch.compile (-O0) and full CUDA graphs for decode,
# captured at every exact batch size 1..max_num_seqs. Both arms get the same
# setting so their performance stays comparable.
VLLM_EAGER = "eager"
VLLM_GRAPH = "graph"
VLLM_EXECUTIONS = (VLLM_EAGER, VLLM_GRAPH)
VLLM_SERVER_ARGS_ENV = "RL_KERNEL_AREAL_VLLM_SERVER_ARGS"
# Not RL_KERNEL_-prefixed, so the scrubbing below keeps them for the native arm too.
PROFILE_STEPS_ENV = "RLK_AREAL_PROFILE_STEPS"
PROFILE_DIR_ENV = "RLK_AREAL_PROFILE_DIR"
NATIVE_ROPE_FUSION_ENV = "RLK_AREAL_NATIVE_ROPE_FUSION"

# Environment that would change numerics; scrubbed from every arm, then the
# RL-Kernel arm sets exactly what it needs.
SCRUBBED_PREFIXES = ("RL_KERNEL_", "VIME_RL_KERNEL_")
SCRUBBED_NAMES = ("VLLM_PLUGINS", "VLLM_ATTENTION_BACKEND", "VLLM_USE_V2_MODEL_RUNNER")


@dataclass(frozen=True)
class ExperimentSpec:
    """Shared workload for both arms; only the arm-specific fields differ."""

    model: str
    mode: str = MODE_SYNC
    steps: int = 5
    seed: int = 1
    train_tp: int = 4
    train_cp: int = 1
    rollout_tp: int = 4
    rollout_dp: int = 1
    n_gpus_per_node: int = 8
    n_nodes: int = 1
    scheduler: str = SCHEDULER_LOCAL
    batch_size: int = 32
    n_samples: int = 4
    max_new_tokens: int = 1024
    max_offpolicyness: int = 1
    lm_head_chunk_size: int = 4096
    max_num_seqs: int = 128
    max_tokens_per_mb: int = 16384
    vllm_execution: str = VLLM_EAGER
    real_vocab_size: int | None = None
    padded_vocab_size: int | None = None
    dataset_path: str | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)
    # Actor versions profiled with torch.profiler on rank 0 (both arms); empty = off.
    profile_steps: tuple[int, ...] = ()
    # Same-version consistency metrics on the native arm in sync mode too (numerics unchanged).
    native_observe: bool = False
    # Megatron TE fused RoPE on the native arm only (numerics unchanged vs stock AReaL intent).
    native_rope_fusion: bool = False

    def validate(self) -> None:
        if self.mode not in (MODE_SYNC, MODE_ASYNC):
            raise ValueError(f"mode must be {MODE_SYNC!r} or {MODE_ASYNC!r}")
        if self.vllm_execution not in VLLM_EXECUTIONS:
            raise ValueError(f"vllm_execution must be one of {VLLM_EXECUTIONS}")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.steps <= 0 or self.batch_size <= 0 or self.n_samples <= 0:
            raise ValueError("steps, batch_size and n_samples must be positive")
        if self.mode == MODE_ASYNC and self.max_offpolicyness <= 0:
            raise ValueError("async mode needs max_offpolicyness >= 1")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}")
        if self.n_nodes <= 0 or self.n_gpus_per_node <= 0:
            raise ValueError("n_nodes and n_gpus_per_node must be positive")
        if any(step < 0 or step >= self.steps for step in self.profile_steps):
            raise ValueError(f"profile_steps must lie in [0, {self.steps})")
        if self.n_nodes > 1 and self.scheduler != SCHEDULER_RAY:
            raise ValueError(
                f"n_nodes={self.n_nodes} needs scheduler={SCHEDULER_RAY!r}; "
                f"the {SCHEDULER_LOCAL!r} scheduler only runs on one node"
            )
        # Tensor/context parallel groups never span nodes: each group must fit in one.
        for name, width in (
            ("train TP*CP", self.train_tp * self.train_cp),
            ("rollout TP", self.rollout_tp),
        ):
            if width > self.n_gpus_per_node:
                raise ValueError(
                    f"{name} ({width}) exceeds the {self.n_gpus_per_node} GPUs of one node"
                )
        used = self.train_tp * self.train_cp + self.rollout_tp * self.rollout_dp
        total = self.n_nodes * self.n_gpus_per_node
        if used > total:
            raise ValueError(
                f"train TP*CP ({self.train_tp}*{self.train_cp}) + rollout TP*DP "
                f"({self.rollout_tp}*{self.rollout_dp}) exceeds {total} GPUs"
            )

    @property
    def vllm_enforce_eager(self) -> bool:
        return self.vllm_execution == VLLM_EAGER

    @property
    def offpolicyness(self) -> int:
        return 0 if self.mode == MODE_SYNC else self.max_offpolicyness

    def actor_backend(self) -> str:
        cp = f"c{self.train_cp}" if self.train_cp > 1 else ""
        return f"megatron:d1p1t{self.train_tp}{cp}"

    def rollout_backend(self) -> str:
        return f"vllm:d{self.rollout_dp}p1t{self.rollout_tp}"


def vllm_server_overrides(spec: ExperimentSpec) -> dict[str, Any]:
    """Extra vLLM server arguments AReaL's ``vLLMConfig`` cannot express.

    AReaL only exposes ``enforce_eager``; ``run_trainer`` appends these to the
    arguments AReaL builds, so they reach ``vllm serve`` unchanged.
    """

    if spec.vllm_execution == VLLM_EAGER:
        return {}
    sizes = list(range(1, spec.max_num_seqs + 1))
    compilation = {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": sizes,
        "max_cudagraph_capture_size": spec.max_num_seqs,
    }
    return {
        "optimization_level": 0,
        "compilation_config": json.dumps(compilation, separators=(",", ":")),
    }


def read_vocabulary(model: str | Path) -> tuple[int, bool]:
    """``(vocab_size, tie_word_embeddings)`` from a local HF checkpoint."""

    config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    vocab = config.get("vocab_size")
    if type(vocab) is not int or vocab <= 0:
        raise ValueError("model config.json must declare a positive integer vocab_size")
    return vocab, bool(config.get("tie_word_embeddings", False))


def _set(node: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _scheduling_spec(base: Any, *, cmd: str, env_vars: Mapping[str, str]) -> list[dict]:
    if isinstance(base, list) and base and isinstance(base[0], dict):
        first = copy.deepcopy(base[0])
    else:
        first = {"task_type": "worker", "port_count": 2, "gpu": 1, "mem": 32}
    first["cmd"] = cmd
    first["env_vars"] = dict(env_vars)
    return [first]


def worker_mode(arm: str, mode: str, native_observe: bool = False) -> str | None:
    """Adapter mode for the actor workers of one arm."""

    if arm == ARM_RLKERNEL:
        return "strict"
    if arm == ARM_NATIVE and (mode == MODE_ASYNC or native_observe):
        return "observe"  # native numerics, same-version metrics only
    return None


def arm_environment(
    spec: ExperimentSpec,
    arm: str,
    *,
    readback_dir: str | Path,
    stats_path: str | Path,
) -> dict[str, str]:
    """Environment variables set for one arm (controller and every worker)."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    env: dict[str, str] = {"RL_KERNEL_AREAL_STATS_JSONL": str(stats_path)}
    if spec.profile_steps:
        env[PROFILE_STEPS_ENV] = ",".join(str(step) for step in sorted(spec.profile_steps))
        env[PROFILE_DIR_ENV] = str(Path(stats_path).parent / "profile")
    overrides = vllm_server_overrides(spec)
    if overrides:
        env[VLLM_SERVER_ARGS_ENV] = json.dumps(overrides, separators=(",", ":"))
    mode = worker_mode(arm, spec.mode, spec.native_observe)
    if mode is not None:
        env["RL_KERNEL_AREAL_MODE"] = mode
    if arm == ARM_NATIVE:
        env["VLLM_PLUGINS"] = ""
        if spec.native_rope_fusion:
            env[NATIVE_ROPE_FUSION_ENV] = "1"
        return env
    if spec.real_vocab_size is None or spec.padded_vocab_size is None:
        raise ValueError("the RL-Kernel arm needs the real and padded vocabulary sizes")
    env.update(
        {
            "RL_KERNEL_ATTENTION_CASE": "R/R",
            "RL_KERNEL_FFN_CASE": "R/R",
            "RL_KERNEL_LOGP_CASE": "R/R",
            "VIME_RL_KERNEL_STRICT": "1",
            "RL_KERNEL_CUDA_ONLY": "1",
            "RL_KERNEL_CUDA_ROPE": "1",
            "RL_KERNEL_VLLM_INTEGRATION": "1",
            "RL_KERNEL_VLLM_REQUIRE_V1": "1",
            "RL_KERNEL_VLLM_REAL_VOCAB_SIZE": str(spec.real_vocab_size),
            "RL_KERNEL_VLLM_PADDED_VOCAB_SIZE": str(spec.padded_vocab_size),
            "RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE": str(spec.max_num_seqs),
            "RL_KERNEL_DET_GEMM_BACKEND": "cublaslt_nosplitk",
            "RL_KERNEL_READBACK_DIR": str(readback_dir),
            "VLLM_PLUGINS": "rl_kernel",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":16:8",
            "CUBLASLT_WORKSPACE_SIZE": "1",
            "NCCL_NVLS_ENABLE": "0",
            # FA4 CuTe JIT-compiles every kernel on first use (the decode kernel
            # alone costs ~6 s of vLLM graph capture). The persistent cache is
            # keyed by the compile key and a fingerprint of the FA4 sources, so
            # later runs load the identical kernel instead of rebuilding it.
            "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
        }
    )
    cache_dir = os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR")
    if cache_dir:
        env["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = cache_dir
    env.update({str(k): str(v) for k, v in spec.extra_env.items()})
    return env


def build_config(
    base: Mapping[str, Any],
    spec: ExperimentSpec,
    arm: str,
    *,
    arm_dir: str | Path,
    env_vars: Mapping[str, str],
    trial_name: str,
) -> dict[str, Any]:
    """Return an edited copy of AReaL's example config for one arm."""

    spec.validate()
    config = copy.deepcopy(dict(base))
    arm_dir = Path(arm_dir)
    settings: dict[str, Any] = {
        "experiment_name": f"rlk-verify-{spec.mode}",
        "trial_name": trial_name,
        "seed": spec.seed,
        "total_train_steps": spec.steps,
        "enable_offload": False,
        "cluster.n_nodes": spec.n_nodes,
        "cluster.n_gpus_per_node": spec.n_gpus_per_node,
        "cluster.fileroot": str(arm_dir / "fileroot"),
        "cluster.name_resolve.type": "nfs",
        "cluster.name_resolve.nfs_record_root": str(arm_dir / "name_resolve"),
        "scheduler.type": spec.scheduler,
        "actor.path": spec.model,
        "actor.backend": spec.actor_backend(),
        "actor.dtype": "bfloat16",
        "actor.disable_dropout": True,
        "actor.kl_ctl": 0.0,
        "actor.recompute_logprob": True,
        "actor.use_decoupled_loss": True,
        "actor.prox_logp_method": "recompute",
        "actor.mb_spec.max_tokens_per_mb": spec.max_tokens_per_mb,
        "actor.megatron.enable_chunked_logits": True,
        "actor.megatron.lm_head_loss_chunk_size": spec.lm_head_chunk_size,
        "actor.megatron.entropy_requires_grad": False,
        "rollout.backend": spec.rollout_backend(),
        "rollout.max_head_offpolicyness": spec.offpolicyness,
        "gconfig.temperature": 1.0,
        "gconfig.top_p": 1.0,
        "gconfig.n_samples": spec.n_samples,
        "gconfig.max_new_tokens": spec.max_new_tokens,
        "train_dataset.batch_size": spec.batch_size,
        "valid_dataset.batch_size": spec.batch_size,
        "vllm.seed": spec.seed,
        "vllm.max_num_seqs": spec.max_num_seqs,
        "vllm.enforce_eager": spec.vllm_enforce_eager,
        "evaluator.freq_epochs": None,
        "evaluator.freq_steps": None,
        "evaluator.freq_secs": None,
        "saver.freq_epochs": None,
        "saver.freq_steps": None,
        "saver.freq_secs": None,
        "recover.mode": "disabled",
        "stats_logger.wandb.mode": "disabled",
        "stats_logger.fileroot": str(arm_dir / "fileroot"),
    }
    if spec.dataset_path:
        settings["train_dataset.path"] = spec.dataset_path
        settings["valid_dataset.path"] = spec.dataset_path
    for dotted, value in settings.items():
        _set(config, dotted, value)
    # The example config has no ref model when kl_ctl == 0; drop it to keep
    # AReaL's shared compute_logp metric free of reference-model scoring.
    config.pop("ref", None)
    mode = worker_mode(arm, spec.mode, spec.native_observe)
    # The profiler is installed by the RL-Kernel worker, which runs AReaL's RPC
    # server unchanged when no adapter mode is selected.
    needs_worker = (
        mode is not None
        or spec.profile_steps
        or (arm == ARM_NATIVE and spec.native_rope_fusion)
    )
    actor_cmd = WORKER_CMD if needs_worker else DEFAULT_RPC_CMD
    actor_base = config.get("actor", {}).get("scheduling_spec")
    config["actor"]["scheduling_spec"] = _scheduling_spec(
        actor_base, cmd=actor_cmd, env_vars=env_vars
    )
    # Rollout workers only launch vLLM; the plugin is selected by environment.
    config["rollout"]["scheduling_spec"] = _scheduling_spec(
        actor_base, cmd=DEFAULT_RPC_CMD, env_vars=env_vars
    )
    return config


def scrubbed_environment(environ: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in environ.items()
        if not key.startswith(SCRUBBED_PREFIXES) and key not in SCRUBBED_NAMES
    }


__all__ = [
    "ARMS",
    "ARM_NATIVE",
    "ARM_RLKERNEL",
    "DEFAULT_CONFIG",
    "DEFAULT_EXAMPLE",
    "DEFAULT_RPC_CMD",
    "PROFILE_DIR_ENV",
    "PROFILE_STEPS_ENV",
    "ExperimentSpec",
    "MODE_ASYNC",
    "MODE_SYNC",
    "SCHEDULERS",
    "SCHEDULER_LOCAL",
    "SCHEDULER_RAY",
    "VLLM_EAGER",
    "VLLM_EXECUTIONS",
    "VLLM_GRAPH",
    "VLLM_SERVER_ARGS_ENV",
    "WORKER_CMD",
    "arm_environment",
    "build_config",
    "read_vocabulary",
    "scrubbed_environment",
    "vllm_server_overrides",
    "worker_mode",
]
