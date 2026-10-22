#!/usr/bin/env python3
"""Gather-gradient reproducibility across ranks: CP-style sharded tokens + all_reduce, and TP-style replicas.

Table [R, D] bf16 (R=42 AdaLN rows), N=32768 tokens, index_select(0, idx) backward, native vs RowGather fix.
- cp: each rank owns N/W tokens; local table grad (bf16) -> all_reduce SUM (bf16) -> compare across 30 repeats / ranks
- tp: every rank holds the full token set (replicated); compare local grads across ranks and repeats (no reduce)
"""
import hashlib, json, os, sys
from pathlib import Path
import torch, torch.distributed as dist
sys.path.insert(0, str(Path(__file__).parent))
from rowgather_fix import patched_index_select

R, D, N, REP = 42, 2048, 32768, 30
rank, world, lr = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]), int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", lr))
dev = torch.device("cuda", lr)
def hh(t): return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]
g = torch.Generator(device=dev).manual_seed(7)
table = (torch.randn(R, D, device=dev, generator=g) * 0.1).bfloat16()
idx = torch.randint(0, R, (N,), device=dev, generator=g)
up = torch.randn(N, D, device=dev, generator=g).bfloat16()
gold = torch.zeros(R, D, dtype=torch.float64, device=dev).index_add_(0, idx, up.double())
sl = slice(rank * N // world, (rank + 1) * N // world)

def local(fix, shard):
    t = table.clone().requires_grad_(True)
    i, u = (idx[sl], up[sl]) if shard else (idx, up)
    def f():
        o = t.index_select(0, i); o.backward(u); return t.grad
    if fix:
        with patched_index_select(): return f()
    return f()

res = {"world": world, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "rows": []}
for layout in ("cp", "tp"):
    for fix in (False, True):
        hs, errs = [], []
        for _ in range(REP):
            gr = local(fix, layout == "cp")
            if layout == "cp": dist.all_reduce(gr)
            hs.append(hh(gr))
            errs.append(float((gr.double() - gold).abs().max() / gold.abs().max()))
        allh = [None] * world; dist.all_gather_object(allh, hs)
        if rank == 0:
            per_rank_distinct = [len(set(x)) for x in allh]
            same_across_ranks = all(allh[r][k] == allh[0][k] for r in range(world) for k in range(REP))
            res["rows"].append({"layout": layout, "impl": "fix" if fix else "native",
                "distinct_per_rank_over_30": per_rank_distinct, "distinct_total": len({h for x in allh for h in x}),
                "ranks_identical_each_repeat": same_across_ranks,
                "err_vs_fp64_min": min(errs), "err_vs_fp64_max": max(errs)})
if rank == 0:
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1)); print(json.dumps(res["rows"], indent=1))
dist.destroy_process_group()
