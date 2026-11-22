#!/usr/bin/env python3
"""Per-row timing on one GPU: h3-parallel-v1's kernel (h3_projection.gemm) vs v2's
(fixed_fp32_gemm) at the tensor-parallel shard shapes, and the pack / noise-mix rows against the
provider's expressions. Median of CUDA-event timings after warm-up. Writes --out (JSON)."""

import argparse
import json
import statistics

import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import fixed_fp32_gemm as fx
from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.backends.shared.triton.gemm import h3_projection as g

ROWS = {"h3_attention_o_gemm": ("row", 7168, 5376), "h3_ffn_down_gemm": ("row", 14336, 5376),
        "h3_ffn_gate_up_gemm": ("col", 5376, 28672)}
KERNELS = {"v1_h3_projection": g.gemm, "v2_fixed_fp32": fx.gemm}


def ms(fn, warm=5, reps=25):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return statistics.median(out)


def gemm_rows(m, tps):
    res = []
    gen = torch.Generator(device="cuda").manual_seed(0)
    for row, (kind, k, n) in ROWS.items():
        x = torch.randn(m, k, device="cuda", generator=gen).bfloat16()
        w = (torch.randn(n, k, device="cuda", generator=gen) / k**0.5).bfloat16()
        grad = torch.randn(m, n, device="cuda", generator=gen).bfloat16()
        for tp in tps:
            for name, K in KERNELS.items():
                if kind == "row":
                    xs = x[:, : k // tp].contiguous(); ws = par.shard_cols(w, tp, 0)
                    t = {"forward": ms(lambda: par._units_partial(xs, ws, tp, kernel=K)),
                         "dx": ms(lambda: K(grad, ws)),
                         "dw": ms(lambda: K(grad, xs, a_t=True))}
                else:
                    ws = par.shard_gate_up(w, tp, 0); gl = grad[:, : n // tp].contiguous()
                    t = {"forward": ms(lambda: K(x, ws, b_t=True)),
                         "dx": ms(lambda: K(gl, ws)),
                         "dw": ms(lambda: K(gl, x, a_t=True))}
                res.append({"row": row, "m": m, "tp": tp, "kernel": name, **t})
    return res


def pack_noise():
    from rl_engine.backends.cuda.conditioning.conditioning_noise_mix import ConditioningNoiseMixCudaOp
    from rl_engine.backends.cuda.packing.packed_index_copy import PackedIndexCopyCudaOp

    gen = torch.Generator(device="cuda").manual_seed(1)
    text, video, audio, d = 512, 3600, 184, 5376
    seq = text + video + audio
    perm = torch.randperm(seq, device="cuda", generator=gen)
    ti, vi, ai = perm[:text], perm[text:text + video], perm[text + video:]
    t, v, a = (torch.randn(1, s, d, device="cuda", generator=gen).bfloat16() for s in (text, video, audio))
    op = PackedIndexCopyCudaOp()

    def provider():
        h = t.new_zeros((1, seq, d))
        return h.index_copy(1, ti, t).index_copy(1, vi, v).index_copy(1, ai, a)

    assert torch.equal(op(t, v, a, ti, vi, ai, seq), provider())
    out = [{"row": "packed_index_copy", "shape": [1, seq, d], "provider_ms": ms(provider),
            "row_ms": ms(lambda: op(t, v, a, ti, vi, ai, seq))}]
    mix = ConditioningNoiseMixCudaOp()
    x = torch.randn(1, 48, 31, 30, 52, device="cuda", generator=gen)
    z = torch.randn(x.shape, device="cuda", generator=gen)
    tt = torch.tensor(0.999, device="cuda")

    def eager():
        return tt * x + (1.0 - tt) * z

    assert torch.equal(mix(x, tt, z), eager())
    out.append({"row": "conditioning_noise_mix", "shape": list(x.shape), "provider_ms": ms(eager),
                "row_ms": ms(lambda: mix(x, tt, z))})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rep = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
           "gemm": [r for m in (928, 4296) for r in gemm_rows(m, (1, 2, 4, 8))], "pack_noise": pack_noise()}
    json.dump(rep, open(args.out, "w"), indent=1)
    print(json.dumps({"ok": True, "n": len(rep["gemm"])}))


if __name__ == "__main__":
    main()
