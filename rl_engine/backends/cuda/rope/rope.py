# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic GPU RoPE ops (GPT-NeoX rotate-half), matching NativeRoPEOp.

cos/sin are built in fp32 with the exact reference math and passed to a small
CUDA kernel (``_C.rope_apply_sm90``) that does the per-position rotation. Backward
reuses the same kernel with the sine negated (RoPE is an orthogonal rotation, so
``grad_x = grad_out * cos - rotate_half(grad_out) * sin``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger


_INV_FREQ: dict[tuple[int, float, torch.device], Tensor] = {}
_TABLES_ATTR = "_rlk_rope_cos_sin"


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _inv_freq(half: int, theta: float, device: torch.device) -> Tensor:
    key = (half, theta, device)
    cached = _INV_FREQ.get(key)
    if cached is not None:
        return cached
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    # A tensor first written inside a CUDA graph capture holds no values until replay.
    if not _capturing():
        _INV_FREQ[key] = inv_freq
    return inv_freq


def share_rope_tables(positions: Tensor) -> Tensor:
    """Let every RoPE call on this ``positions`` object reuse one cos/sin table.

    Only for position tensors that are never written again while referenced:
    the table is keyed by object identity, not by content.
    """

    if not torch.compiler.is_compiling():
        setattr(positions, _TABLES_ATTR, {})
    return positions


def _build_cos_sin(positions: Tensor, half: int, theta: float, device: torch.device):
    """fp32 cos/sin caches of shape [S, half], identical math to NativeRoPEOp."""
    tables = getattr(positions, _TABLES_ATTR, None)
    key = (half, theta, device)
    if tables is not None and key in tables:
        return tables[key]
    inv_freq = _inv_freq(half, theta, device)
    pos = positions.to(device=device, dtype=torch.float32).reshape(-1, 1)
    freqs = pos * inv_freq  # [S, half]
    result = freqs.cos().contiguous(), freqs.sin().contiguous()
    if tables is not None:
        tables[key] = result
    return result


def _rope_table(x: Tensor, positions: Tensor, theta: float) -> tuple[Tensor, Tensor, Tensor]:
    """Build (x_2d, cos, sin) for [S] or [B, S] positions. See Triton RoPE."""
    D = x.shape[-1]
    if D % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {D}")
    if positions.dim() == 1:
        table_len = int(positions.shape[0])
        x_2d = x.contiguous().reshape(-1, D)
        if x_2d.shape[0] % table_len != 0:
            raise ValueError(
                f"row count {x_2d.shape[0]} not divisible by seq length {table_len}; "
                "expected a [..., S, D] contiguous layout."
            )
        cos, sin = _build_cos_sin(positions, D // 2, float(theta), x.device)
        return x_2d, cos, sin
    if positions.dim() != 2:
        raise ValueError(f"positions must be [S] or [B, S], got shape {tuple(positions.shape)}")
    batch, seq = positions.shape
    if x.shape[0] != batch or x.shape[-2] != seq:
        raise ValueError(
            f"positions {tuple(positions.shape)} is incompatible with x {tuple(x.shape)}; "
            "expected x [B, ..., S, D]"
        )
    if x.dim() == 4:
        x_2d = x.permute(1, 0, 2, 3).contiguous().reshape(-1, D)
    elif x.dim() == 3:
        x_2d = x.contiguous().reshape(-1, D)
    else:
        raise ValueError(
            f"RoPE [B, S] positions require x [B, S, D] or [B, H, S, D], got {x.dim()}D"
        )
    table_len = batch * seq
    if x_2d.shape[0] % table_len != 0:
        raise ValueError(f"row count {x_2d.shape[0]} not divisible by B*S={table_len}")
    cos, sin = _build_cos_sin(positions.reshape(-1), D // 2, float(theta), x.device)
    return x_2d, cos, sin


def _restore_rope(out_2d: Tensor, x: Tensor, positions: Tensor) -> Tensor:
    if positions.dim() == 1 or x.dim() != 4:
        return out_2d.reshape(x.shape)
    heads, batch, seq, dim = x.shape[1], x.shape[0], x.shape[2], x.shape[3]
    return out_2d.reshape(heads, batch, seq, dim).permute(1, 0, 2, 3).contiguous()


class _RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, positions: Tensor, theta: float) -> Tensor:
        x_2d, cos, sin = _rope_table(x, positions, theta)
        ctx.save_for_backward(cos, sin)
        ctx.x_shape = tuple(x.shape)
        ctx.pos_dim = positions.dim()
        out_2d = _rope_apply(x_2d, cos, sin, 1.0)
        return _restore_rope(out_2d, x, positions)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cos, sin = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            if ctx.pos_dim == 2 and len(ctx.x_shape) == 4:
                g_2d = grad_out.permute(1, 0, 2, 3).contiguous().reshape(-1, ctx.x_shape[-1])
                out_2d = _rope_apply(g_2d, cos, sin, -1.0)
                heads, batch, seq, dim = (
                    ctx.x_shape[1],
                    ctx.x_shape[0],
                    ctx.x_shape[2],
                    ctx.x_shape[3],
                )
                grad_x = out_2d.reshape(heads, batch, seq, dim).permute(1, 0, 2, 3).contiguous()
            else:
                g_2d = grad_out.contiguous().reshape(-1, grad_out.shape[-1])
                grad_x = _rope_apply(g_2d, cos, sin, -1.0).reshape(grad_out.shape)
        return grad_x, None, None


def _rope_apply(x: Tensor, cos: Tensor, sin: Tensor, sin_sign: float) -> Tensor:
    if torch.version.hip is not None:
        return _C.deterministic_rope_apply_rocm(x, cos, sin, sin_sign)
    return _C.rope_apply_sm90(x, cos, sin, sin_sign)


_IS_HOPPER: dict[torch.device, bool] = {}


def _is_hopper(device: torch.device) -> bool:
    cached = _IS_HOPPER.get(device)
    if cached is not None:
        return cached
    try:
        value = torch.cuda.get_device_capability(device)[0] == 9
    except Exception:
        return False
    _IS_HOPPER[device] = value
    return value


def _token_major_cos_sin(
    positions: Tensor, heads: int, half: int, theta: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Per-row tables for a token-major ``[T * H, D]`` view: row ``t * H + h`` uses ``t``."""

    tables = getattr(positions, _TABLES_ATTR, None)
    key = ("token_major", heads, half, theta, device)
    if tables is not None and key in tables:
        return tables[key]
    cos, sin = _build_cos_sin(positions, half, theta, device)
    result = cos.repeat_interleave(heads, dim=0), sin.repeat_interleave(heads, dim=0)
    if tables is not None:
        tables[key] = result
    return result


class RoPESM90Op:
    """Custom CUDA RoPE op for SM90 (GPT-NeoX rotate-half), differentiable w.r.t. ``x``.

    Qwen3 defaults: theta=1e6, head_dim=128, full-dimension rotation. cos/sin are
    computed in fp32 from ``positions`` and ``theta`` (matching the reference); the
    rotation runs in the precompiled ``_C.rope_apply_sm90`` CUDA kernel.
    """

    op_class = "elementwise"

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "rope_apply_sm90"):
            raise RuntimeError(
                "CUDA RoPE kernel 'rope_apply_sm90' is not compiled into _C. "
                "Rebuild the extension with 'KERNEL_ALIGN_FORCE_SM90=1 pip install -e .'."
            )
        logger.info("Successfully linked to precompiled _C.rope_apply_sm90 kernel.")

    def __call__(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        return self.forward(x, positions, theta=theta)

    def forward(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        if x.device.type != "cuda":
            raise RuntimeError(f"RoPESM90Op requires a CUDA tensor, got device '{x.device}'.")
        if not _is_hopper(x.device):
            raise RuntimeError(
                "RoPESM90Op requires Hopper (SM90) CUDA; "
                f"got compute capability {torch.cuda.get_device_capability(x.device)}"
            )
        return _RoPEFunction.apply(x, positions, theta)

    def token_major(
        self, value: Tensor, positions: Tensor, head_dim: int, *, theta: float = 1_000_000.0
    ) -> Tensor | None:
        """Rotate ``[T, H * D]`` in place of layout, or None when autograd is needed.

        The kernel is a per-row rotation, so feeding token-major rows with a table
        repeated per head computes exactly what the head-major path computes,
        without its two transposed copies per tensor.
        """

        if torch.is_grad_enabled() and value.requires_grad:
            return None
        if value.device.type != "cuda" or not _is_hopper(value.device):
            return None
        if positions.dim() != 1 or value.dim() != 2 or value.shape[0] != positions.shape[0]:
            return None
        x_2d = value.contiguous().reshape(-1, head_dim)
        cos, sin = _token_major_cos_sin(
            positions, value.shape[1] // head_dim, head_dim // 2, float(theta), value.device
        )
        return _rope_apply(x_2d, cos, sin, 1.0).reshape(value.shape)


class RocmDeterministicRoPEOp:
    """Precompiled HIP RoPE path shared by ROCm training and rollout."""

    backend_id = "rlkernel.rocm.deterministic_rope"
    op_class = "elementwise"
    fallback = False

    def __init__(self) -> None:
        if torch.version.hip is None:
            raise RuntimeError("RocmDeterministicRoPEOp requires a ROCm PyTorch build")
        if not _EXT_AVAILABLE or not hasattr(_C, "deterministic_rope_apply_rocm"):
            raise RuntimeError(
                "ROCm deterministic RoPE is unavailable; rebuild rl_engine._C for ROCm"
            )

    def __call__(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        return self.forward(x, positions, theta=theta)

    def forward(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        if not x.is_cuda:
            raise RuntimeError("ROCm deterministic RoPE requires a GPU tensor")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("ROCm deterministic RoPE requires FP16 or BF16")
        return _RoPEFunction.apply(x, positions, theta)


__all__ = ["RoPESM90Op", "RocmDeterministicRoPEOp", "share_rope_tables"]
