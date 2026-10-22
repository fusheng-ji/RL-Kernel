#!/usr/bin/env python3
"""Deterministic, FP32-accumulated backward for ``Tensor.index_select(0, idx)``.

``RowGather.apply(table, idx)``: forward == ``table.index_select(0, idx)`` (bitwise);
backward: stable sort of idx, FP32 segmented sum (``torch.segment_reduce``, fixed order), cast back to
the table dtype. No atomics, so the result is bitwise repeatable.

``patched_index_select()`` swaps ``Tensor.index_select`` for dim 0 / floating tensors that require grad
inside a ``with`` block, so unmodified diffusers code uses it.
"""
from __future__ import annotations

import contextlib

import torch


class RowGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, idx):
        ctx.save_for_backward(idx)
        ctx.rows, ctx.dtype = table.shape[0], table.dtype
        return _orig(table, 0, idx)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        order = torch.sort(idx, stable=True).indices
        lengths = torch.bincount(idx, minlength=ctx.rows)
        flat = g.reshape(g.shape[0], -1)
        data = _orig(flat, 0, order).float()
        out = torch.segment_reduce(data, "sum", lengths=lengths, axis=0)
        return out.to(ctx.dtype).reshape(ctx.rows, *g.shape[1:]), None


_orig = torch.Tensor.index_select


@contextlib.contextmanager
def patched_index_select():
    def fn(self, dim, index):
        if dim == 0 and self.requires_grad and self.is_floating_point() and index.dim() == 1:
            return RowGather.apply(self, index)
        return _orig(self, dim, index)

    torch.Tensor.index_select = fn
    try:
        yield
    finally:
        torch.Tensor.index_select = _orig
