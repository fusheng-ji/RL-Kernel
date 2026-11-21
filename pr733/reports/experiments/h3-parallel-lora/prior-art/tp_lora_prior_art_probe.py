"""The cross-rank sums of tensor-parallel LoRA as peft / Megatron and vLLM do them, against TP = 1.

peft's ``LoraParallelLinear`` (Megatron backend) and vLLM's ``*ParallelLinearWithLoRA`` split A and B
as h3-parallel-lora-v1 does: on a row-parallel base A is cut by input columns and B is whole; on a
column-parallel base A is whole and B is cut by output rows. The partial products are then summed
across ranks by an all-reduce. This probe forms those partial products with torch GEMMs, sums them in
rank order (one valid all-reduce order; NCCL's ring order is another), and compares with the TP = 1
product, for the two sums that cross ranks:

- row-parallel forward ``z = x A^T`` (attention output, K = 7168; ``ff.net.2``, K = 14336);
- column-parallel backward ``dz = dy B`` (Q, N = 7168; ``ff.net.0.proj``, N = 28672).

Adapters and activations in BF16 (the recipe's dtype, as peft computes them) and in FP32.
Same seeded LoRA as the harness (r = 64, seed 7, B random), 256 rows.

    python tp_lora_prior_art_probe.py --out report.json
"""
import argparse, json
import torch
from rl_engine.validation.models import h3_lora

CASES = [("attn.to_out.0", "row: z = x A^T"), ("ff.net.2", "row: z = x A^T"),
         ("attn.to_q", "col: dz = dy B"), ("ff.net.0.proj", "col: dz = dy B")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--rows", type=int, default=256)
    args = ap.parse_args()
    lora = h3_lora.make_block_lora(0, seed=7, device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    rows = []
    for i, (target, what) in enumerate(CASES):
        a, b = lora[target]["A"], lora[target]["B"]
        gen = torch.Generator().manual_seed(100 + i)
        if what.startswith("row"):
            k = a.shape[1]; inp = torch.randn(args.rows, k, generator=gen).cuda(); w = a  # (r, K): z = inp @ w^T
            part = lambda t, w, lo, hi: t[:, lo:hi] @ w[:, lo:hi].t()
        else:
            n = b.shape[0]; inp = torch.randn(args.rows, n, generator=gen).cuda(); w = b.t()  # dz = dy @ B = dy @ (B^T)^T
            part = lambda t, w, lo, hi: t[:, lo:hi] @ w[:, lo:hi].t()
        width = inp.shape[1]
        for dtype in (torch.bfloat16, torch.float32):
            t, ww = inp.to(dtype), w.to(dtype)
            ref = part(t, ww, 0, width)
            for tp in (2, 4, 8):
                step = width // tp
                acc = part(t, ww, 0, step)
                for r in range(1, tp):
                    acc = acc + part(t, ww, r * step, (r + 1) * step)
                differ = int((acc != ref).sum().item())
                rows.append({"target": target, "sum": what, "dtype": str(dtype).replace("torch.", ""), "tp": tp,
                             "elements": ref.numel(), "differ_from_tp1": differ,
                             "max_abs": float((acc.float() - ref.float()).abs().max().item())})
                print(json.dumps(rows[-1]))
    rep = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "rows_per_case": args.rows,
           "lora": {"rank": h3_lora.RANK, "seed": 7, "b": "random"}, "cases": rows}
    open(args.out, "w").write(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
