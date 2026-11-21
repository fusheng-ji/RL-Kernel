"""Does the gradient norm depend on how the gradients are partitioned? torch vs exact_norm.

The gradients are LoRA-shaped FP32 tensors (rank 32 on the 5376-wide H3 projections, 50 blocks
x 7 projections x A/B). Each "layout" splits every tensor into k contiguous row shards, as TP
does, and optionally permutes the tensor order. torch's way (torch.nn.utils.get_total_norm on
each partition, then sqrt of the sum of squares across partitions, which is what FSDP2/DTensor
and Megatron's float all-reduce compute) is compared bitwise with the k=1 result; exact_norm
of the same shards with that of the whole tensors. Single process, one GPU.

    python norm_partition_probe.py --out report.json
"""
import argparse, json
import torch
from rl_engine.ops.norm.reproducible import exact_norm


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    g = torch.Generator(device="cpu").manual_seed(0)
    grads = []
    for _ in range(50 * 7):
        grads.append((torch.randn(32, 5376, generator=g) * 1e-3).cuda())   # A
        grads.append((torch.randn(5376, 32, generator=g) * 1e-3).cuda())   # B
    def torch_norm(parts):
        local = [torch.nn.utils.get_total_norm(p, 2.0) for p in parts]
        return float(torch.sqrt(sum(l.float() ** 2 for l in local)))
    ref_torch = float(torch.nn.utils.get_total_norm(grads, 2.0)); ref_exact = exact_norm(grads)
    rows = []
    for k in (1, 2, 4, 8):
        for perm_seed in (None, 1, 2, 3):
            order = list(range(len(grads))) if perm_seed is None else torch.randperm(len(grads), generator=torch.Generator().manual_seed(perm_seed)).tolist()
            ts = [grads[i] for i in order]
            parts = [[t.chunk(k, dim=0)[r] for t in ts] for r in range(k)]
            tn = torch_norm(parts); en = exact_norm([s for p in parts for s in p])
            rows.append({"partitions": k, "permutation": perm_seed, "torch": tn, "torch_equals_unpartitioned": tn == ref_torch,
                         "exact": en, "exact_equals_unpartitioned": en == ref_exact})
    rep = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "tensors": len(grads),
           "elements": sum(t.numel() for t in grads), "unpartitioned_torch": ref_torch, "unpartitioned_exact": ref_exact,
           "layouts": len(rows), "torch_bitwise_equal": sum(r["torch_equals_unpartitioned"] for r in rows),
           "exact_bitwise_equal": sum(r["exact_equals_unpartitioned"] for r in rows),
           "torch_values": sorted({r["torch"] for r in rows}), "rows": rows}
    open(args.out, "w").write(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("torch", "tensors", "elements", "layouts", "torch_bitwise_equal", "exact_bitwise_equal", "torch_values", "unpartitioned_exact")}))


if __name__ == "__main__":
    main()
