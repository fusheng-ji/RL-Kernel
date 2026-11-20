"""Prior art: vLLM 0.33's batch-invariant mean_dim on the H3 log-prob rows vs h3-row-mean-v1.

Same inputs as row_mean_probe.py (a batch of 8 rows; video K = 3,096,576, audio K = 6,624):
time per call, relative error against FP64, batch invariance (each row alone vs in batches of
2 / 4 / 8) and whether a row at another memory offset gives the same bytes. One GPU.

    python vllm_mean_probe.py --out report.json
"""
import argparse, json, statistics
import torch
from vllm.model_executor.determinism.batch_invariant import mean_dim
from rl_engine.backends.shared.triton.reduction import row_mean

SIZES = {"video": 24 * 32 * 48 * 84, "audio": 207 * 32}
FNS = {"vllm_mean_dim": lambda x: mean_dim(x, 1), "row_mean_v1": lambda x: row_mean.row_sums(x) / x.shape[1]}


def timed(fn, repeats=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    rows = []
    for mod, k in SIZES.items():
        gen = torch.Generator().manual_seed(1)
        x = (-(torch.randn(8, k, generator=gen) * 0.1) ** 2).cuda(); ref = x.double().mean(dim=1)
        big = torch.empty(8 * k + 16, device="cuda"); shifted = big[3 : 3 + 8 * k].view(8, k); shifted.copy_(x)
        for name, fn in FNS.items():
            m = fn(x); alone = [fn(x[i : i + 1]) for i in range(8)]; differ = 0
            for b in (2, 4, 8):
                for s in range(0, 8, b):
                    out = fn(x[s : s + b])
                    differ += sum(int(not torch.equal(out[j : j + 1].view(torch.int32), alone[s + j].view(torch.int32))) for j in range(b))
            rows.append({"modality": mod, "K": k, "mean": name,
                         "max_rel_err_vs_fp64": ((m.double() - ref).abs() / ref.abs()).max().item(),
                         "differ_from_alone_of_24": differ,
                         "offset_rows_equal": int(torch.equal(fn(shifted).view(torch.int32), m.view(torch.int32))),
                         "ms_B1": timed(lambda: fn(x[:1])), "ms_B8": timed(lambda: fn(x))})
    rep = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "rows": rows}
    open(args.out, "w").write(json.dumps(rep, indent=1) + "\n")
    for r in rows: print(json.dumps({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in r.items()}))


if __name__ == "__main__":
    main()
