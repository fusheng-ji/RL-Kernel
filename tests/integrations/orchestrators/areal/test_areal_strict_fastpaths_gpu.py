# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bitwise checks for the strict fast paths on an SM90 GPU."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] != 9,
    reason="strict CUDA fast paths require an SM90 GPU",
)


def _lengths() -> list[int]:
    return [57, 640, 57, 1031, 3, 640, 640, 2900, 128, 129]


def _packed_qkv(lengths: list[int], q_heads: int, kv_heads: int, head_dim: int):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    tokens = sum(lengths)
    qkv = torch.randn(
        tokens,
        q_heads + 2 * kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    dout = torch.randn(
        tokens,
        q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return qkv, dout


def test_packed_varlen_attention_matches_grouped_strict_runtime():
    pytest.importorskip("flash_attn.cute.interface")
    from rl_engine.backends.cuda.attention.strict_runtime import StrictCUDAAttentionRuntime
    from rl_engine.contracts.operators.attention import AttentionRole
    from rl_engine.integrations.common.operators import _dense_attention_contract

    q_heads, kv_heads, head_dim = 8, 2, 128
    scale = head_dim**-0.5
    lengths = _lengths()
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    qkv0, dout = _packed_qkv(lengths, q_heads, kv_heads, head_dim)
    runtime = StrictCUDAAttentionRuntime()

    def split(qkv):
        return (
            qkv[:, :q_heads],
            qkv[:, q_heads : q_heads + kv_heads],
            qkv[:, q_heads + kv_heads :],
        )

    grouped_qkv = qkv0.clone().requires_grad_()
    query, key, value = split(grouped_qkv)
    groups: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        groups.setdefault(length, []).append(index)
    outputs: list[torch.Tensor | None] = [None] * len(lengths)
    grouped_lse: list[torch.Tensor | None] = [None] * len(lengths)
    for length, indices in groups.items():
        ready = [
            torch.stack(
                [tensor[offsets[i] : offsets[i + 1]].permute(1, 0, 2) for i in indices]
            )
            for tensor in (query, key, value)
        ]
        positions = torch.arange(length, device="cuda").repeat(len(indices), 1)
        result = runtime.forward_with_lse(
            *ready,
            contract=_dense_attention_contract(
                ready[0],
                ready[1],
                role=AttentionRole.TRAIN,
                causal=True,
                tp_rank=0,
                tp_world_size=1,
            ),
            causal=True,
            scale=scale,
            cp_world_size=1,
            query_position_ids=positions,
            key_position_ids=positions,
        )
        out = result.out.permute(0, 2, 1, 3).contiguous()
        for batch, index in enumerate(indices):
            outputs[index] = out[batch]
            grouped_lse[index] = result.lse[batch]
    grouped_out = torch.cat(outputs)
    grouped_out.backward(dout)

    packed_qkv = qkv0.clone().requires_grad_()
    query, key, value = split(packed_qkv)
    packed = runtime.forward_packed_with_lse(
        query,
        key,
        value,
        cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device="cuda"),
        max_seqlen=max(lengths),
        causal=True,
        scale=scale,
    )
    packed.out.backward(dout)

    assert torch.equal(packed.out, grouped_out)
    for index, lse in enumerate(grouped_lse):
        assert torch.equal(packed.lse[:, offsets[index] : offsets[index + 1]], lse)
    assert torch.equal(packed_qkv.grad, grouped_qkv.grad)
    assert packed.provenance["core_launch_count"] == 1
    assert packed.provenance["core_actual_backends"] == ["flash_attention_4.cute.varlen"]


def test_shared_rope_tables_match_per_call_tables_on_gpu():
    from rl_engine.backends.cuda.rope.rope import RoPESM90Op, share_rope_tables

    op = RoPESM90Op()
    positions = torch.cat([torch.arange(n, device="cuda") for n in (17, 300, 5)])
    generator = torch.Generator(device="cuda").manual_seed(7)
    query = torch.randn(positions.numel(), 8, 128, device="cuda", dtype=torch.bfloat16,
                        generator=generator)
    key = torch.randn(positions.numel(), 2, 128, device="cuda", dtype=torch.bfloat16,
                      generator=generator)
    grad = torch.randn_like(query)

    def run(shared: bool):
        q = query.clone().requires_grad_()
        k = key.clone().requires_grad_()
        pos = share_rope_tables(positions.view(-1)) if shared else positions
        q_out = op(q, pos)
        k_out = op(k, pos)
        (q_out.float() * grad.float()).sum().add(k_out.float().sum()).backward()
        return q_out, k_out, q.grad, k.grad

    for plain, shared in zip(run(False), run(True), strict=True):
        assert torch.equal(plain, shared)


@pytest.mark.parametrize("tokens", [1, 7, 128, 2048])
def test_token_major_rope_matches_head_major_rope(tokens):
    from rl_engine.backends.cuda.rope.rope import RoPESM90Op, share_rope_tables
    from rl_engine.integrations.common.cuda_rope import token_major_rope

    op = RoPESM90Op()
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    positions = torch.randint(0, 4000, (tokens,), device="cuda", generator=generator)
    query = torch.randn(tokens, 32 * 128, device="cuda", dtype=torch.bfloat16,
                        generator=generator)
    key = torch.randn(tokens, 8, 128, device="cuda", dtype=torch.bfloat16, generator=generator)

    def head_major(value):
        flat = value.reshape(tokens, -1)
        heads = flat.reshape(tokens, -1, 128).transpose(0, 1).contiguous()
        out = op(heads, positions)
        return out.transpose(0, 1).contiguous().reshape(value.shape)

    with torch.inference_mode():
        shared = share_rope_tables(positions.view(-1))
        for value in (query, key):
            fast = token_major_rope(value, shared, 128, op)
            assert fast.shape == value.shape
            assert torch.equal(fast, head_major(value))


def test_identity_temperature_logp_matches_unit_temperature_tensor(tmp_path):
    import torch.distributed as dist

    from rl_engine.backends.cuda.logprob.linear_logp import (
        sm90_deterministic_logp_from_local_logits_tp,
    )

    created = not dist.is_initialized()
    if created:
        dist.init_process_group(
            "nccl", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1
        )
    try:
        group = dist.new_group([0])
        generator = torch.Generator(device="cuda").manual_seed(3)
        rows, vocab = 96, 4096
        logits0 = torch.randn(rows, vocab, device="cuda", dtype=torch.bfloat16,
                              generator=generator)
        targets = torch.randint(0, vocab, (rows,), device="cuda", generator=generator)
        upstream = torch.randn(rows, device="cuda", generator=generator)

        def run(temperature):
            logits = logits0.clone().requires_grad_()
            logp, lse = sm90_deterministic_logp_from_local_logits_tp(
                logits,
                targets,
                vocab_start_index=0,
                global_vocab_size=vocab,
                real_vocab_size=vocab,
                temperature=temperature,
                tp_group=group,
            )
            (logp.float() * upstream).sum().backward()
            return logp, lse, logits.grad

        skipped = run(None)
        unit = run(torch.ones(rows, device="cuda", dtype=torch.float32))
        for left, right in zip(skipped, unit, strict=True):
            assert torch.equal(left, right)
    finally:
        if created:
            dist.destroy_process_group()
