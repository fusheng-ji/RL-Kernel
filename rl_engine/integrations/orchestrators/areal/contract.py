# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Supported AReaL configuration boundary for the strict train/rollout route.

Everything here is framework-free: it reads attributes from AReaL objects
without importing AReaL, so the checks can be unit-tested on CPU. Unsupported
configurations are rejected; the adapter never falls back to native numerics
while still reporting a strict run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

MODE_ENV = "RL_KERNEL_AREAL_MODE"
MODE_STRICT = "strict"
MODE_OBSERVE = "observe"
MODES = (MODE_STRICT, MODE_OBSERVE)

REAL_VOCAB_ENV = "RL_KERNEL_VLLM_REAL_VOCAB_SIZE"
PADDED_VOCAB_ENV = "RL_KERNEL_VLLM_PADDED_VOCAB_SIZE"

SUPPORTED_TENSOR_PARALLEL = (1, 2, 4, 8)
SUPPORTED_CONTEXT_PARALLEL = (1, 2)
QWEN3_ARCHITECTURE = "Qwen3ForCausalLM"


def adapter_mode(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the requested adapter mode, or ``None`` when the adapter is off."""

    env = os.environ if environ is None else environ
    value = env.get(MODE_ENV, "").strip().lower()
    if not value:
        return None
    if value not in MODES:
        raise ValueError(f"{MODE_ENV} must be one of {MODES}; got {value!r}")
    return value


def vocabulary_contract(environ: Mapping[str, str] | None = None) -> tuple[int, int]:
    """Real and padded vocabulary shared by training and rollout."""

    env = os.environ if environ is None else environ
    values = []
    for name in (REAL_VOCAB_ENV, PADDED_VOCAB_ENV):
        raw = env.get(name, "").strip()
        if not raw.isdecimal() or int(raw) <= 0:
            raise ValueError(f"strict AReaL training requires a positive {name}")
        values.append(int(raw))
    real, padded = values
    if real > padded:
        raise ValueError(f"{REAL_VOCAB_ENV}={real} exceeds {PADDED_VOCAB_ENV}={padded}")
    return real, padded


def _flag(owner: Any, name: str) -> bool:
    return bool(getattr(owner, name, False))


def engine_config_errors(engine: Any) -> list[str]:
    """Errors visible before ``MegatronEngine.initialize`` builds anything."""

    errors: list[str] = []
    config = getattr(engine, "config", None)
    mcore = getattr(engine, "mcore_config", None)
    if config is None or mcore is None:
        return ["engine has no config/mcore_config"]
    if getattr(config, "dtype", None) != "bfloat16":
        errors.append("actor.dtype must be bfloat16")
    if _flag(config, "is_critic"):
        errors.append("critic engines are unsupported")
    if _flag(config, "use_lora"):
        errors.append("LoRA is unsupported")
    if _flag(engine, "enable_tree_training") or _flag(config, "enable_tree_training"):
        errors.append("tree training is unsupported")
    if not _flag(config, "disable_dropout"):
        errors.append("actor.disable_dropout must be true")
    temperature = getattr(config, "temperature", 1.0)
    if temperature is None or float(temperature) != 1.0:
        errors.append("actor.temperature must be 1.0")
    if getattr(engine, "bridge_cls", getattr(mcore, "bridge_type", "mbridge")) != "mbridge":
        errors.append("megatron.bridge_type must be mbridge")
    if not _flag(mcore, "enable_chunked_logits"):
        errors.append("megatron.enable_chunked_logits must be true")
    if int(getattr(mcore, "lm_head_loss_chunk_size", 0) or 0) <= 0:
        errors.append("megatron.lm_head_loss_chunk_size must be positive")
    for name in (
        "entropy_requires_grad",
        "enable_fp32_lm_head",
        "enable_mtp",
        "enable_mtp_training",
        "mtp_only",
        "use_torch_fsdp2",
        "use_custom_fsdp",
        "enable_router_replay",
    ):
        if _flag(mcore, name):
            errors.append(f"megatron.{name} is unsupported")
    if getattr(mcore, "fp8_config", None) is not None:
        errors.append("FP8 training is unsupported")
    if (getattr(mcore, "virtual_pipeline_parallel_size", None) or 1) > 1:
        errors.append("virtual pipeline parallelism is unsupported")
    strategy = getattr(engine, "parallel_strategy", None)
    if strategy is not None:
        errors.extend(parallel_strategy_errors(strategy))
    return errors


def parallel_strategy_errors(strategy: Any) -> list[str]:
    errors: list[str] = []
    tp = int(getattr(strategy, "tensor_parallel_size", 1) or 1)
    cp = int(getattr(strategy, "context_parallel_size", 1) or 1)
    pp = int(getattr(strategy, "pipeline_parallel_size", 1) or 1)
    ep = int(getattr(strategy, "expert_parallel_size", 1) or 1)
    if tp not in SUPPORTED_TENSOR_PARALLEL:
        errors.append(f"tensor_parallel_size={tp} is not in {SUPPORTED_TENSOR_PARALLEL}")
    if cp not in SUPPORTED_CONTEXT_PARALLEL:
        errors.append(f"context_parallel_size={cp} is not in {SUPPORTED_CONTEXT_PARALLEL}")
    if pp != 1:
        errors.append("pipeline parallelism is unsupported")
    if ep != 1:
        errors.append("expert parallelism is unsupported")
    return errors


def _rope_config_errors(hf_config: Any) -> list[str]:
    """Accept equivalent HF default-RoPE schemas without enabling scaled RoPE."""

    errors: list[str] = []
    theta_values = []
    theta = getattr(hf_config, "rope_theta", None)
    if theta is not None:
        theta_values.append(theta)
    for name in ("rope_scaling", "rope_parameters"):
        config = getattr(hf_config, name, None)
        if config is None:
            continue
        if not isinstance(config, Mapping):
            errors.append(f"{name} must be a default RoPE mapping or None")
            continue
        if not config:
            continue
        types = [config[key] for key in ("rope_type", "type") if key in config]
        if not types or any(value != "default" for value in types):
            errors.append(f"{name}: only default RoPE is supported")
        unknown = set(config) - {"rope_type", "type", "rope_theta"}
        if unknown:
            errors.append(f"{name}: unsupported RoPE fields {sorted(unknown)}")
        if "rope_theta" in config:
            theta_values.append(config["rope_theta"])
    for value in theta_values:
        try:
            valid = not isinstance(value, bool) and float(value) == 1_000_000.0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            errors.append("strict CUDA Qwen3 RoPE requires theta=1000000")
            break
    # When HF omits theta, the existing Megatron RotaryEmbedding constructor
    # guard still checks the effective rotary_base before any kernel executes.
    return errors


def model_config_errors(engine: Any, hf_config: Any, tf_config: Any) -> list[str]:
    """Errors visible once the HF and Megatron configurations exist."""

    errors: list[str] = []
    architectures = list(getattr(hf_config, "architectures", None) or [])
    if architectures != [QWEN3_ARCHITECTURE] or getattr(hf_config, "model_type", None) != "qwen3":
        errors.append(f"only dense Qwen3 is supported; got {architectures}")
    if getattr(hf_config, "quantization_config", None):
        errors.append("quantized checkpoints are unsupported")
    if _flag(hf_config, "tie_word_embeddings"):
        errors.append("tied embeddings are unsupported by the strict LM head")
    errors.extend(_rope_config_errors(hf_config))
    if _flag(engine, "use_padded_seq") or _flag(engine, "is_vision_model"):
        errors.append("padded BSHD / vision forwards are unsupported")
    if _flag(tf_config, "sequence_parallel"):
        errors.append("sequence parallelism is still enabled")
    if _flag(tf_config, "apply_rope_fusion"):
        errors.append("fused RoPE is still enabled")
    if _flag(tf_config, "use_mup"):
        errors.append("MuP output scaling is unsupported")
    return errors


def raise_if(errors: list[str], where: str) -> None:
    if errors:
        raise ValueError(f"RL-Kernel AReaL {where}: " + "; ".join(errors))


__all__ = [
    "MODES",
    "MODE_ENV",
    "MODE_OBSERVE",
    "MODE_STRICT",
    "PADDED_VOCAB_ENV",
    "REAL_VOCAB_ENV",
    "adapter_mode",
    "engine_config_errors",
    "model_config_errors",
    "parallel_strategy_errors",
    "raise_if",
    "vocabulary_contract",
]
