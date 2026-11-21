# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Opt-in shared CUDA RoPE for dense Qwen3 training (Megatron) and rollout (vLLM).

Megatron and vLLM ship different RoPE implementations, so post-RoPE Q/K can
differ in the last bit even when every other operator is aligned. With
``RL_KERNEL_CUDA_ROPE=1`` both frameworks call the same ``RoPESM90Op`` with
token-major positions. This module only maps framework layouts and positions;
the numerical work stays in the existing kernel.

The first call in each process writes a small marker into
``RL_KERNEL_READBACK_DIR`` so launchers can prove the route actually ran.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch

ENV_FLAG = "RL_KERNEL_CUDA_ROPE"
QWEN3_THETA = 1_000_000.0
_VLLM_MARKER = "__rl_kernel_original_cuda_rope_forward__"
_MEGATRON_MARKER = "__rl_kernel_original_cuda_rope_apply__"
_CALLS: dict[str, int] = {"training": 0, "rollout": 0}


def enabled() -> bool:
    """CUDA builds only; ROCm already shares ``RocmDeterministicRoPEOp``."""

    return torch.version.hip is None and os.getenv(ENV_FLAG, "").strip() == "1"


def call_counts() -> dict[str, int]:
    return dict(_CALLS)


def _record_call(role: str, backend: str) -> None:
    _CALLS[role] += 1
    if _CALLS[role] != 1:
        return
    print(f"[RL-Kernel RoPE] {role} backend={backend} theta={QWEN3_THETA:g}", flush=True)
    directory = os.getenv("RL_KERNEL_READBACK_DIR", "").strip()
    if not directory:
        return
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload = {"module": "rope", "role": role, "backend_id": backend, "first_call": True}
    path = target / f"cuda-rope-{role}-{os.getpid()}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def token_major_rope(
    value: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    operator: Any,
    theta: float = QWEN3_THETA,
) -> torch.Tensor:
    """Apply a head-major RoPE operator to vLLM's ``[T, H*D]`` (or ``[T, H, D]``) layout."""

    original_shape = value.shape
    if value.ndim == 3:
        value = value.reshape(value.shape[0], -1)
    if value.ndim != 2 or value.shape[1] % head_dim:
        raise ValueError("CUDA RoPE expects [tokens, heads * head_dim]")
    flat_positions = positions.reshape(-1)
    if flat_positions.numel() != value.shape[0]:
        raise ValueError("CUDA RoPE position count differs from token count")
    heads = value.reshape(value.shape[0], -1, head_dim).transpose(0, 1).contiguous()
    result = operator(heads, flat_positions, theta=theta)
    return result.transpose(0, 1).contiguous().reshape(original_shape)


def logical_positions(freqs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Read logical positions from Megatron's frequency table.

    Channel zero of an unscaled rotate-half table is ``position * theta**0``, so it
    equals the logical position even when a packed table resets per sequence.
    Table indices alone are not position IDs in that case.
    """

    channel_zero = freqs[:, 0, 0, 0].float()
    flat = indices.reshape(-1).to(device=channel_zero.device, dtype=torch.int64)
    values = channel_zero.index_select(0, flat)
    positions = torch.round(values).to(torch.int64)
    if not torch.equal(positions.to(values.dtype), values):
        raise RuntimeError("strict CUDA RoPE frequency table is not integer-position aligned")
    return positions.reshape(indices.shape).to(indices.device)


def install_vllm(rotary_cls: type[Any]) -> None:
    """Route vLLM's unscaled rotate-half ``RotaryEmbedding.forward_cuda`` to RoPESM90Op."""

    if hasattr(rotary_cls, _VLLM_MARKER):
        return
    from rl_engine.backends.cuda.rope.rope import RoPESM90Op

    operator = RoPESM90Op()
    original = rotary_cls.forward_cuda

    @functools.wraps(original)
    def forward_cuda(instance: Any, positions: Any, query: Any, key: Any = None, *args, **kwargs):
        if args or any(value is not None for value in kwargs.values()):
            raise ValueError("strict CUDA RoPE does not support offsets or extra arguments")
        # Subclasses may change frequency scaling without overriding this method.
        if type(instance) is not rotary_cls:
            raise ValueError("strict CUDA RoPE does not support scaled rotary subclasses")
        head_dim = int(instance.head_size)
        if head_dim != int(instance.rotary_dim) or not bool(instance.is_neox_style):
            raise ValueError("strict CUDA RoPE requires full-dimension rotate-half")
        base = getattr(instance, "base", None)
        if base is None or float(base) != QWEN3_THETA:
            raise ValueError("strict CUDA Qwen3 RoPE requires theta=1000000")
        if not torch.compiler.is_compiling():
            _record_call("rollout", "rlkernel.cuda.rope_sm90")
        rotated_query = token_major_rope(query, positions, head_dim, operator)
        rotated_key = None if key is None else token_major_rope(key, positions, head_dim, operator)
        return rotated_query, rotated_key

    setattr(rotary_cls, _VLLM_MARKER, original)
    rotary_cls.forward_cuda = forward_cuda


def _context_parallel_group(cp_group: Any) -> Any:
    if cp_group is not None:
        return cp_group
    try:
        from megatron.core import parallel_state

        if parallel_state.get_context_parallel_world_size() > 1:
            return parallel_state.get_context_parallel_group()
    except (ImportError, AssertionError, RuntimeError):
        pass
    return None


def install_megatron() -> None:
    """Route Megatron's unfused ``apply_rotary_pos_emb`` to RoPESM90Op.

    Fused RoPE (``apply_rope_fusion=True``) bypasses this function; callers must
    disable it before model construction.
    """

    attention = importlib.import_module("megatron.core.transformer.attention")
    if hasattr(attention, _MEGATRON_MARKER):
        return
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    from rl_engine.backends.cuda.rope.rope import RoPESM90Op
    from rl_engine.integrations.engines.train.megatron.runtime import (
        _apply_strict_rocm_rope,
        _strict_rocm_rope_positions,
    )

    operator = RoPESM90Op()
    original_init = RotaryEmbedding.__init__
    signature = inspect.signature(original_init)
    state = {"validated": False}

    @functools.wraps(original_init)
    def init(instance: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(instance, *args, **kwargs)
        bound.apply_defaults()
        cfg = bound.arguments
        if (
            float(cfg.get("rotary_base", 10000)) != QWEN3_THETA
            or float(cfg.get("rotary_percent", 1.0)) != 1.0
            or bool(cfg.get("rotary_interleaved", False))
            or bool(cfg.get("rope_scaling", False))
            or cfg.get("seq_len_interpolation_factor") is not None
        ):
            raise ValueError("strict CUDA RoPE requires unscaled full Qwen3 theta=1000000")
        original_init(instance, *args, **kwargs)
        state["validated"] = True

    original_apply = attention.apply_rotary_pos_emb

    @functools.wraps(original_apply)
    def apply(
        tensor: torch.Tensor,
        freqs: torch.Tensor,
        config: Any,
        cu_seqlens: torch.Tensor | None = None,
        mscale: float = 1.0,
        cp_group: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if any(value is not None for value in kwargs.values()):
            raise ValueError(f"strict CUDA RoPE does not support {sorted(kwargs)}")
        if not state["validated"]:
            raise RuntimeError("strict CUDA RoPE did not validate the model rotary configuration")
        if (
            bool(getattr(config, "rotary_interleaved", False))
            or bool(getattr(config, "multi_latent_attention", False))
            or float(mscale) != 1.0
        ):
            raise ValueError("unsupported strict CUDA RoPE layout or scaling")
        if freqs.ndim != 4 or tuple(freqs.shape[1:3]) != (1, 1):
            raise ValueError("strict CUDA RoPE requires [S, 1, 1, head_dim] frequencies")
        if freqs.shape[-1] != tensor.shape[-1]:
            raise ValueError("strict CUDA RoPE requires full-dimension frequencies")
        indices = _strict_rocm_rope_positions(
            tensor, freqs, cu_seqlens, _context_parallel_group(cp_group)
        )
        positions = logical_positions(freqs, indices)
        _record_call("training", "rlkernel.cuda.rope_sm90")
        return _apply_strict_rocm_rope(tensor, positions, operator)

    RotaryEmbedding.__init__ = init
    setattr(attention, _MEGATRON_MARKER, original_apply)
    attention.apply_rotary_pos_emb = apply


__all__ = [
    "ENV_FLAG",
    "QWEN3_THETA",
    "call_counts",
    "enabled",
    "install_megatron",
    "install_vllm",
    "logical_positions",
    "token_major_rope",
]
