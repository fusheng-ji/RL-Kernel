# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict drop-in for AReaL's ``chunked_lm_head_logprobs_entropy``.

AReaL's chunked LM-head path already owns every layout decision: it rolls
next-token labels before context-parallel splitting, and reassembles and
unpads the returned per-token scalars itself. This replacement keeps that
contract (same arguments, same ``ChunkedLMHeadOutput`` fields) and only swaps
the numerics: the deterministic TP output projection plus the shared strict
``linear_logp`` provider that the vLLM rollout side also uses.

The four vocabulary statistics are computed over the local TP shard, exactly
as AReaL's own implementation does; they are diagnostics, not inputs to the
loss.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

BACKEND_ID = "rlkernel.areal.strict_lm_head"


def _tp_group(output_layer: Any) -> Any:
    group = getattr(output_layer, "tp_group", None)
    if group is not None:
        return group
    from megatron.core import parallel_state

    return parallel_state.get_tensor_model_parallel_group()


def _group_coordinates(group: Any) -> tuple[int, int]:
    if group is None:
        return 0, 1
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return int(dist.get_rank(group=group)), int(dist.get_world_size(group=group))


def _cp_coordinates() -> tuple[int, int]:
    try:
        from megatron.core import parallel_state

        return (
            int(parallel_state.get_context_parallel_rank()),
            int(parallel_state.get_context_parallel_world_size()),
        )
    except (ImportError, AssertionError, RuntimeError):
        return 0, 1


def validate_head_call(
    output_layer: Any,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    logit_scale: float,
) -> None:
    """Reject every case whose numerics this head does not reproduce."""

    if float(temperature) != 1.0:
        raise ValueError("strict AReaL LM head requires temperature=1.0")
    if float(logit_scale) != 1.0:
        raise ValueError("strict AReaL LM head does not support MuP logit scaling")
    for name in ("sequence_parallel", "gather_output"):
        if bool(getattr(output_layer, name, False)):
            raise ValueError(f"strict AReaL LM head does not support {name}")
    if getattr(output_layer, "bias", None) is not None:
        raise ValueError("strict AReaL LM head requires a bias-free Qwen3 head")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError(
            "strict AReaL LM head requires BF16 hidden states and weight; got "
            f"{hidden_states.dtype} and {weight.dtype}"
        )
    if weight.ndim != 2 or hidden_states.shape[-1] != weight.shape[1]:
        raise ValueError("hidden size does not match the LM-head weight")
    rows = hidden_states.numel() // hidden_states.shape[-1]
    if rows != labels.numel():
        raise ValueError(
            "LM-head hidden leading dimensions must match labels: "
            f"{tuple(hidden_states.shape)} vs {tuple(labels.shape)}"
        )


def make_strict_chunked_head(output_cls: Any, *, real_vocab_size: int, padded_vocab_size: int):
    """Build the replacement bound to AReaL's ``ChunkedLMHeadOutput`` type."""

    def strict_chunked_lm_head_logprobs_entropy(
        output_layer: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        *,
        temperature: float,
        chunk_size: int,
        logit_scale: float = 1.0,
    ) -> Any:
        del chunk_size  # The strict route materializes one vocabulary shard per row.
        validate_head_call(
            output_layer,
            hidden_states,
            weight,
            labels,
            temperature=temperature,
            logit_scale=logit_scale,
        )
        from rl_engine.integrations.engines.train.megatron.runtime import (
            _DeterministicTPOutputProjection,
        )
        from rl_engine.integrations.orchestrators.vime.providers.linear_logp_provider import (
            provider,
        )

        tp_group = _tp_group(output_layer)
        tp_rank, tp_world = _group_coordinates(tp_group)
        cp_rank, cp_world = _cp_coordinates()
        local_size = int(weight.shape[0])
        if local_size * tp_world != padded_vocab_size:
            raise ValueError(
                "training LM-head shards do not match the rollout vocabulary contract: "
                f"{local_size} * TP{tp_world} != {padded_vocab_size}"
            )
        # Megatron [S, B, H] activations become S-major rows, matching labels.
        local_logits = _DeterministicTPOutputProjection.apply(
            hidden_states, weight, None, tp_group
        )
        logits_2d = local_logits.reshape(-1, local_size)
        hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        targets = labels.reshape(-1).to(torch.int64)
        request = SimpleNamespace(
            logits=logits_2d,
            target_ids=targets,
            tensor_parallel_group=tp_group,
            token_layout=SimpleNamespace(
                world_size=cp_world,
                rank=cp_rank,
                layout="single" if cp_world == 1 else "zigzag",
            ),
            with_entropy=True,
            with_entropy_grad=False,
            temperature=1.0,
            chunk_size=0,
            log_prob_keep_mask=None,
            context=SimpleNamespace(
                hidden=hidden_2d,
                projection=SimpleNamespace(weight=weight, bias=None),
                vocab_partition=SimpleNamespace(
                    local_start=tp_rank * local_size,
                    local_size=local_size,
                    real_size=real_vocab_size,
                    padded_size=padded_vocab_size,
                ),
                reuse_local_logits=True,
                local_logits=logits_2d,
            ),
            metadata={"tp_rank": tp_rank, "tp_world_size": tp_world},
        )
        result = provider(request)
        if tuple(result.logp.shape) != (targets.numel(), 1):
            raise RuntimeError("strict provider returned misaligned logp rows")
        if not result.provenance.get("execution", {}).get("lm_head_result_reused"):
            raise RuntimeError("strict provider did not score the current LM-head result")
        if torch.is_grad_enabled() and logits_2d.requires_grad and not result.logp.requires_grad:
            raise RuntimeError("strict provider detached differentiable logp")
        if result.entropy is None or result.entropy.numel() != targets.numel():
            raise RuntimeError("strict provider did not return token-aligned metric entropy")
        shape = labels.shape
        with torch.no_grad():
            detached = logits_2d.detach()
            stats = (
                detached.min(dim=-1).values.float().reshape(shape),
                detached.max(dim=-1).values.float().reshape(shape),
                detached.mean(dim=-1, dtype=torch.float32).reshape(shape),
                torch.linalg.vector_norm(detached, dim=-1, dtype=torch.float32).reshape(shape),
            )
        return output_cls(
            result.logp.reshape(shape).float(),
            result.entropy.detach().reshape(shape).float(),
            *stats,
        )

    strict_chunked_lm_head_logprobs_entropy.backend_id = BACKEND_ID  # type: ignore[attr-defined]
    return strict_chunked_lm_head_logprobs_entropy


__all__ = ["BACKEND_ID", "make_strict_chunked_head", "validate_head_call"]
