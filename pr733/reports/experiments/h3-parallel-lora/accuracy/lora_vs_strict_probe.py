"""Canonical h3-parallel-lora-v1 (tp=1) against #574's strict_lora_linear and FP64, per target.

Real block-0 weights of the released checkpoint (``RL_KERNEL_H3_WEIGHTS``), the seeded LoRA of
the harness (r = 64, alpha = 128, B random), 256 BF16 rows of random activations and output
gradients. For each of the six targets (FFN-in as the fused [hidden, gate] weight): relative
error of y, dX, dA and dB against FP64 for both, and whether y is bitwise strict_lora_linear's.

    python lora_vs_strict_probe.py --out report.json
"""
import argparse, json, subprocess
import torch
from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl
from rl_engine.models.minimax_h3.runtime import strict_lora_linear
from rl_engine.validation.models import h3_lora
from rl_engine.validation.models.h3_weights import load_h3_block_weights

WEIGHT = {"attn.to_q": "attn.to_q.weight", "attn.to_k": "attn.to_k.weight", "attn.to_v": "attn.to_v.weight",
          "attn.to_out.0": "attn.to_out.0.weight", "ff.net.0.proj": "ff.net.0.proj.weight", "ff.net.2": "ff.net.2.weight"}


def leaf(t):
    return t.detach().clone().requires_grad_(True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--rows", type=int, default=256)
    args = ap.parse_args()
    weights = load_h3_block_weights("cuda")
    key = lambda n: next(k for k in weights if k.endswith(n))
    lora = h3_lora.make_block_lora(0, seed=7, device="cuda")
    scale = h3_lora.ALPHA / h3_lora.RANK
    rows = []
    for target, kind in pl.TARGETS.items():
        w = weights[key(WEIGHT[target])].bfloat16().contiguous()
        n, k = w.shape
        gen = torch.Generator().manual_seed(len(rows) + 1)
        x = torch.randn(args.rows, k, generator=gen).bfloat16().cuda()
        dy = torch.randn(args.rows, n, generator=gen).bfloat16().cuda()
        a, b = lora[target]["A"], lora[target]["B"]
        # ours, canonical (tp = 1): with the BF16 A / B of the recipe, and with FP32 A / B of the same
        # values (the arithmetic alone: gradients are then not rounded to BF16)
        out = {}
        for pdt in ("bf16", "fp32"):
          ap, bp = (a, b) if pdt == "bf16" else (a.float(), b.float())
          xs, al, bl = leaf(x), leaf(ap), leaf(bp)
          if kind == "col":
              if target == "ff.net.0.proj":
                  half = n // 2
                  bh, bg = leaf(bp[:half]), leaf(bp[half:])
                  yh, yg = pl.column_parallel_lora_linear(xs, [w[:half].contiguous(), w[half:].contiguous()], [(al, [bh, bg])], scale)
                  y = torch.cat([yh, yg], 1); torch.autograd.backward([yh, yg], [dy[:, :half], dy[:, half:]])
                  db = torch.cat([bh.grad, bg.grad])
              else:
                  (y,) = pl.column_parallel_lora_linear(xs, [w], [(al, [bl])], scale); y.backward(dy); db = bl.grad
          else:
              y = pl.row_parallel_lora_linear(xs, w, al, bl, scale); y.backward(dy); db = bl.grad
          out[pdt] = (y.detach(), xs.grad, al.grad, db)
        ours = out["bf16"]
        # #574 strict LoRA (FP32 A / B of the same values)
        xs2, a2, b2 = leaf(x), leaf(a.float()), leaf(b.float())
        y2 = strict_lora_linear(xs2, w, a2, b2, scale); y2.backward(dy)
        strict = (y2.detach(), xs2.grad, a2.grad, b2.grad)
        # FP64
        x64, w64, a64, b64, dy64 = (t.double() for t in (x, w, a, b, dy))
        z = x64 @ a64.t(); dz = scale * dy64 @ b64
        ref = (x64 @ w64.t() + scale * z @ b64.t(), dy64 @ w64 + dz @ a64, dz.t() @ x64, scale * dy64.t() @ z)
        rel = lambda t, r: ((t.double() - r).norm() / r.norm()).item()
        row = {"target": target, "kind": kind, "K": k, "N": n, "y_bitwise_strict": bool(torch.equal(ours[0], strict[0]))}
        for name, o, o32, s, r in zip(("y", "dx", "dA", "dB"), ours, out["fp32"], strict, ref):
            row[f"{name}_ours_bf16_params"] = rel(o, r); row[f"{name}_ours_fp32_params"] = rel(o32, r)
            row[f"{name}_strict"] = rel(s, r)
        rows.append(row)
        print(json.dumps({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in row.items()}))
    rep = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "rows_per_case": args.rows,
           "rl_kernel_commit": subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "lora": {"rank": h3_lora.RANK, "alpha": h3_lora.ALPHA, "seed": 7, "b": "random"}, "targets": rows}
    open(args.out, "w").write(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
