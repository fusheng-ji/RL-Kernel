"""fixed_order_mean before (fixed-order GEMM, h3-projection-gemm-v1) and after (h3-row-mean-v1).

For the H3 latent sizes (video 24 x 32 x 48 x 84 = 3,096,576 per sample, audio 207 x 32 = 6,624):
time per call (CUDA events, 30 calls after 5 warm-up), error against an FP64 mean, and
whether each sample's mean is the same bytes alone and in batches of 2, 4 and 8. One GPU.

    python row_mean_probe.py --out report.json
"""
import argparse, json, statistics
import torch
from rl_engine.backends.shared.triton.gemm import h3_projection as g
from rl_engine.backends.shared.triton.reduction import row_mean

SIZES = {"video": 24 * 32 * 48 * 84, "audio": 207 * 32}


def old_mean(flat):
    ones = torch.ones(flat.shape[1], 1, device=flat.device, dtype=torch.float32)
    return g.gemm(flat, ones)[:, 0] / flat.shape[1]


def new_mean(flat):
    return row_mean.row_sums(flat) / flat.shape[1]


def timed(fn, repeats=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return {"median_ms": statistics.median(ts), "min_ms": min(ts)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    rows = []
    for mod, k in SIZES.items():
        gen = torch.Generator().manual_seed(1)
        # a log-prob-like input: -(x_prev - mean)^2 with std ~ 0.1 noise
        x = (-(torch.randn(8, k, generator=gen) * 0.1) ** 2).cuda()
        ref = x.double().mean(dim=1)
        for name, fn in (("gemm_h3_projection_v1", old_mean), ("row_mean_v1", new_mean)):
            m = fn(x)
            rel = ((m.double() - ref).abs() / ref.abs()).max().item()
            alone = torch.stack([fn(x[i : i + 1].contiguous())[0] for i in range(8)])
            differ = 0; total = 0
            for b in (2, 4, 8):
                for s in range(0, 8, b):
                    out = fn(x[s : s + b].contiguous())
                    for j in range(b):
                        total += 1; differ += int(not torch.equal(out[j].view(torch.int32), alone[s + j].view(torch.int32)))
            t1 = timed(lambda: fn(x[:1].contiguous())); t2 = timed(lambda: fn(x[:2].contiguous())); t8 = timed(lambda: fn(x))
            rows.append({"modality": mod, "K": k, "mean": name, "max_rel_err_vs_fp64": rel, "batch_comparisons": total,
                         "differ_from_alone": differ, "ms_B1": t1, "ms_B2": t2, "ms_B8": t8})
    rep = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "rows": rows}
    open(args.out, "w").write(json.dumps(rep, indent=1) + "\n")
    for r in rows: print(json.dumps({k: (round(v["median_ms"], 3) if isinstance(v, dict) else v) for k, v in r.items()}))


if __name__ == "__main__":
    main()
