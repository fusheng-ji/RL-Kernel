"""fix/vllm-sort-topk-topp-contract evidence: vLLM's Triton top-k/top-p against
the sort that sampling_keep_mask reproduces.

    PYTHONPATH=<branch worktree> python topp_contract_bench.py <out.json>
"""
import json
import sys

import torch
import vllm
from vllm.v1.sample.ops import topk_topp_sampler as ops

from rl_engine.ops.sampling.policy import sampling_keep_mask

VOCAB = 151936
triton_dispatch = ops.apply_top_k_top_p  # read before any RL-Kernel install


def masks(rows, top_p, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, VOCAB, device="cuda", generator=g)
    p = torch.full((rows,), top_p, device="cuda")
    policy = sampling_keep_mask(logits, temperature=0.7, top_p=p)
    tri = torch.isfinite(triton_dispatch(logits.clone() / 0.7, None, p))
    srt = torch.isfinite(ops.apply_top_k_top_p_pytorch(logits.clone() / 0.7, None, p))
    return logits, p, policy, tri, srt


def timed(fn, iters=50, repeats=5):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)
    return sorted(times)[len(times) // 2]


def main():
    support = []
    for top_p in (0.8, 0.9, 0.95, 0.99):
        logits, p, policy, tri, srt = masks(64, top_p, seed=int(top_p * 1000))
        probs = torch.where(tri, logits / 0.7, float("-inf")).softmax(-1)
        extra = tri & ~policy
        support.append({
            "rows": 64,
            "top_p": top_p,
            "sort_equals_policy": bool(torch.equal(srt, policy)),
            "triton_superset_of_policy": bool(torch.equal(tri & policy, policy)),
            "rows_differing_triton": int((tri != policy).any(1).sum()),
            "extra_tokens_per_row_max": int(extra.sum(1).max()),
            "extra_tokens_total": int(extra.sum()),
            "triton_only_prob_mass_mean": float((probs * extra).sum(1).mean()),
            "triton_only_prob_mass_max": float((probs * extra).sum(1).max()),
        })
        print(json.dumps(support[-1]), flush=True)
    # Batch dependence: row r alone and in a batch of 8 against the batch of 64.
    logits, p, policy, tri, srt = masks(64, 0.95, seed=7)
    variant = {"triton": 0, "sort": 0, "cases": 0}
    for r in range(64):
        for n in (1, 8):
            sub = logits[r : r + n].clone() / 0.7
            m = sub.size(0)
            pp = p[:m]
            variant["triton"] += int(not torch.equal(torch.isfinite(triton_dispatch(sub.clone(), None, pp))[0], tri[r]))
            variant["sort"] += int(not torch.equal(torch.isfinite(ops.apply_top_k_top_p_pytorch(sub.clone(), None, pp))[0], srt[r]))
            variant["cases"] += 1
    print(json.dumps(variant), flush=True)
    timing = []
    for rows in (1, 8, 64, 256):
        x = torch.randn(rows, VOCAB, device="cuda")
        p = torch.full((rows,), 0.95, device="cuda")
        timing.append({
            "rows": rows,
            "triton_ms": timed(lambda: triton_dispatch(x.clone(), None, p)),
            "sort_ms": timed(lambda: ops.apply_top_k_top_p_pytorch(x.clone(), None, p)),
            "clone_ms": timed(lambda: x.clone()),
        })
        print(json.dumps(timing[-1]), flush=True)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "vocab": VOCAB,
        "temperature": 0.7,
        "support": support,
        "batch_dependence": variant,
        "timing": timing,
    }
    with open(sys.argv[1], "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
