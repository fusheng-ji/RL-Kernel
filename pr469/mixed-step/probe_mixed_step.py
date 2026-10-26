#!/usr/bin/env python3
"""Does a prefill sharing the step change a Qwen3-Next GDN decode row in vLLM 0.30.0?

Kernel-level replay of the two paths ``QwenGatedDeltaNetAttention._forward_core`` takes for the same decode rows
(vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py):

  A  decode-only step, packed fast path (VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1, the default):
       causal_conv1d_update -> fused_recurrent_gated_delta_rule_packed_decode
  B  decode-only step, packed path off:
       causal_conv1d_update -> fused_sigmoid_gating_delta_rule_update
  C  mixed step (decodes + one prefill, decodes peeled off by ``split_non_spec``):
       causal_conv1d_fn over [decode tokens | prefill tokens] -> fused_sigmoid_gating_delta_rule_update on the decode rows

It calls the library kernels with synthetic inputs (no checkpoint, no attention metadata, no scheduler), so it shows
what the kernels do, not what a served model does. Prints one JSON document on stdout.
"""
import json
import sys

import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)

K = V = 128
W = 4
CONFIGS = {"tp1": (16, 32), "tp4_per_rank": (4, 8)}  # (key heads, value heads) per rank


def make(H, HV, B, seed, state_dtype, num_blocks):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s, dt: torch.randn(*s, device="cuda", dtype=dt, generator=g)  # noqa: E731
    dim = 2 * H * K + HV * V
    return {
        "dim": dim,
        "x": r(B, dim, dt=torch.bfloat16),
        "a": r(B, HV, dt=torch.bfloat16),
        "b": r(B, HV, dt=torch.bfloat16),
        "A_log": r(HV, dt=torch.float32),
        "dt_bias": r(HV, dt=torch.float32),
        "w": r(dim, W, dt=torch.bfloat16),
        "conv": r(num_blocks, dim, W - 1, dt=torch.bfloat16),
        "ssm": r(num_blocks, HV, V, K, dt=state_dtype) * 0.1,
        "idx": torch.arange(1, B + 1, device="cuda", dtype=torch.int32),
        "prefill_x": r(4096, dim, dt=torch.bfloat16),
        "prefill_a": r(4096, HV, dt=torch.bfloat16),
        "prefill_b": r(4096, HV, dt=torch.bfloat16),
    }


def split_qkv(x, H, HV):
    q, k, v = torch.split(x, [H * K, H * K, HV * V], dim=-1)
    t = x.shape[0]
    return (q.reshape(1, t, H, K).contiguous(), k.reshape(1, t, H, K).contiguous(), v.reshape(1, t, HV, V).contiguous())


def path_a(d, H, HV):
    conv, ssm = d["conv"].clone(), d["ssm"].clone()
    B = d["x"].shape[0]
    y = causal_conv1d_update(d["x"].clone(), conv, d["w"], None, "silu", conv_state_indices=d["idx"], validate_data=False)
    out = torch.empty(B, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=y, a=d["a"], b=d["b"], A_log=d["A_log"], dt_bias=d["dt_bias"], scale=K**-0.5,
        initial_state=ssm, out=out, ssm_state_indices=d["idx"], use_qk_l2norm_in_kernel=True,
    )
    return {"conv_out": y, "out": out.reshape(B, HV, V), "ssm": ssm[d["idx"].long()], "conv": conv[d["idx"].long()]}


def recurrent_nonpacked(d, H, HV, y, ssm):
    B = y.shape[0]
    q, k, v = split_qkv(y, H, HV)
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=q, k=k, v=v, initial_state=ssm,
        inplace_final_state=True, cu_seqlens=torch.arange(B + 1, device="cuda", dtype=torch.int32),
        ssm_state_indices=d["idx"], use_qk_l2norm_in_kernel=True,
    )
    return o.reshape(B, HV, V)


def path_b(d, H, HV):
    conv, ssm = d["conv"].clone(), d["ssm"].clone()
    y = causal_conv1d_update(d["x"].clone(), conv, d["w"], None, "silu", conv_state_indices=d["idx"], validate_data=False)
    out = recurrent_nonpacked(d, H, HV, y, ssm)
    return {"conv_out": y, "out": out, "ssm": ssm[d["idx"].long()], "conv": conv[d["idx"].long()]}


def path_c(d, H, HV, prefill_len):
    conv, ssm = d["conv"].clone(), d["ssm"].clone()
    B = d["x"].shape[0]
    L = prefill_len
    xs = torch.cat([d["x"], d["prefill_x"][:L]], dim=0)  # decode tokens first
    qsl = torch.tensor(list(range(B)) + [B, B + L], device="cuda", dtype=torch.int32)
    cache_idx = torch.cat([d["idx"], torch.tensor([B + 1], device="cuda", dtype=torch.int32)])
    has_init = torch.tensor([True] * B + [False], device="cuda")
    y_all = causal_conv1d_fn(
        xs.transpose(0, 1), d["w"], None, conv_states=conv, query_start_loc=qsl, cache_indices=cache_idx,
        has_initial_state=has_init, activation="silu",
    ).transpose(0, 1)
    y = y_all[:B].contiguous()
    out = recurrent_nonpacked(d, H, HV, y, ssm)
    return {"conv_out": y, "out": out, "ssm": ssm[d["idx"].long()], "conv": conv[d["idx"].long()]}


def diff(p, q):
    r = {}
    for k in ("conv_out", "out", "ssm", "conv"):
        a, b = p[k], q[k]
        ne = (a != b)
        per_row = ne.reshape(ne.shape[0], -1).sum(1)
        r[k] = {
            "elements": int(a.numel()),
            "differing": int(ne.sum()),
            "rows_with_a_difference": int((per_row > 0).sum()),
            "rows": int(a.shape[0]),
            "max_per_row_differing": int(per_row.max()),
            "elements_per_row": int(a.numel() // a.shape[0]),
            "max_abs_diff": float((a.float() - b.float()).abs().max()),
        }
    return r


def main():
    res = {"vllm": __import__("vllm").__version__, "torch": torch.__version__, "device": torch.cuda.get_device_name(), "runs": []}
    for cfg, (H, HV) in CONFIGS.items():
        for state_dtype in (torch.float32, torch.bfloat16):
            for B in (1, 4, 17, 64):
                d = make(H, HV, B, seed=B, state_dtype=state_dtype, num_blocks=B + 2)
                a, b = path_a(d, H, HV), path_b(d, H, HV)
                c1, c2 = path_c(d, H, HV, 64), path_c(d, H, HV, 1031)
                res["runs"].append({
                    "config": cfg, "key_heads": H, "value_heads": HV, "state_dtype": str(state_dtype).split(".")[-1], "decode_rows": B,
                    "A_packed_vs_B_nonpacked": diff(a, b),
                    "B_nonpacked_vs_C_mixed": diff(b, c1),
                    "A_packed_vs_C_mixed": diff(a, c1),
                    "C_prefill64_vs_C_prefill1031": diff(c1, c2),
                    "repeat_A_bitwise": all(torch.equal(a[k], path_a(d, H, HV)[k]) for k in a),
                })
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
