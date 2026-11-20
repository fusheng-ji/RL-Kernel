"""Gradient norm and clip on real TP / TP x CP layouts: torch DTensor vs RL-Kernel exact.

torchrun --nproc-per-node 8 multirank_norm_probe.py --out report.json

The gradients are the LoRA-shaped FP32 tensors of the partition probe (700 tensors, rank 32 on
the 5376-wide projections). Layouts: one GPU (reference), TP2, TP4, TP8 (rows sharded over
TP), TP4 x CP2 (sharded over TP, replicated over CP), one torchrun per world size (2, 4, 8); each run covers the layouts of its world.
torch: DTensor gradients on a DeviceMesh (Shard(0) over tp, Replicate over cp) and
torch.nn.utils.clip_grad_norm_, what FSDP2 / TP training calls. RL-Kernel: exact_norm /
clip_grad_norm_exact_ on the local shards, counted once per CP group (cp rank 0).
For each layout: the norm (bitwise vs that implementation's one-GPU norm, and equal on all
ranks) and the clipped gradients gathered back (bytes vs the one-GPU clip).
"""
import argparse, json, os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from rl_engine.ops.norm.reproducible import clip_grad_norm_exact_, exact_norm

MAX_NORM = 1.0


def grads_cpu():
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(*s, generator=g) * 1e-3) for s in [(32, 5376), (5376, 32)] * 350]


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(-1).view(torch.int32), b.contiguous().view(-1).view(torch.int32))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); dev = torch.device("cuda")
    full = [t.to(dev) for t in grads_cpu()]
    # one-GPU references (every rank computes them; they must agree)
    ref = {}
    ps = [torch.nn.Parameter(torch.zeros_like(t)) for t in full]
    for p, t in zip(ps, full): p.grad = t.clone()
    ref["torch_norm"] = float(torch.nn.utils.clip_grad_norm_(ps, MAX_NORM)); ref["torch_clipped"] = [p.grad for p in ps]
    ps = [torch.nn.Parameter(torch.zeros_like(t)) for t in full]
    for p, t in zip(ps, full): p.grad = t.clone()
    alone = [dist.new_group([r]) for r in range(world)][rank]  # a one-rank group: no all-reduce
    ref["exact_norm"] = clip_grad_norm_exact_(ps, MAX_NORM, group=alone); ref["exact_clipped"] = [p.grad for p in ps]
    layouts = [("TP2", 2, 1), ("TP4", 4, 1), ("TP8", 8, 1), ("TP4xCP2", 4, 2)]
    rows = []
    for name, tp, cp in layouts:
        n = tp * cp
        if n != world: continue
        sub = None
        active = True
        row = {"layout": name, "ranks": n}
        if active:
            mesh = init_device_mesh("cuda", (cp, tp), mesh_dim_names=("cp", "tp"))
            cp_rank, tp_rank = rank // tp, rank % tp
            # torch DTensor path (needs the mesh to span the whole world: layouts on all 8 ranks)
            if mesh is not None:
                params = []
                for t in full:
                    p = torch.nn.Parameter(distribute_tensor(torch.zeros_like(t), mesh, [Replicate(), Shard(0)]))
                    p.grad = distribute_tensor(t.clone(), mesh, [Replicate(), Shard(0)])
                    params.append(p)
                tn = torch.nn.utils.clip_grad_norm_(params, MAX_NORM)
                tn = float(tn.full_tensor() if hasattr(tn, "full_tensor") else tn)
                clipped = [p.grad.full_tensor() for p in params]
                row.update(torch_norm=tn, torch_norm_equal=tn == ref["torch_norm"],
                           torch_clipped_equal=all(same(a, b) for a, b in zip(clipped, ref["torch_clipped"])))
            # exact path: local row shards; replicated over CP, counted on cp rank 0
            ps = [torch.nn.Parameter(torch.zeros_like(t.chunk(tp)[tp_rank])) for t in full]
            for p, t in zip(ps, full): p.grad = t.chunk(tp)[tp_rank].clone()
            en = clip_grad_norm_exact_(ps, MAX_NORM, counted=lambda p: cp_rank == 0, group=sub)
            shards = [p.grad for p in ps]
            ok = True
            for s, r in zip(shards, ref["exact_clipped"]):
                ok &= same(s, r.chunk(tp)[tp_rank])
            row.update(exact_norm=en, exact_norm_equal=en == ref["exact_norm"], exact_clipped_equal_local=bool(ok))
        gathered = [None] * world
        dist.all_gather_object(gathered, row if active else None)
        if rank == 0:
            rs = [r for r in gathered if r is not None]
            agg = {"layout": name, "ranks": n,
                   "exact_norm_equal_all_ranks": all(r["exact_norm_equal"] for r in rs),
                   "exact_norm_values": sorted({r["exact_norm"] for r in rs}),
                   "exact_clipped_equal": all(r["exact_clipped_equal_local"] for r in rs)}
            if "torch_norm" in rs[0]:
                agg.update(torch_norm_values=sorted({r["torch_norm"] for r in rs}), torch_norm_equal_all_ranks=all(r["torch_norm_equal"] for r in rs),
                           torch_clipped_equal=all(r["torch_clipped_equal"] for r in rs))
            else:
                agg.update(torch_norm_values=None, torch_note="DTensor mesh spans the whole world; this layout uses part of it")
            rows.append(agg)
        dist.barrier()
    if rank == 0:
        rep = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "world": world, "max_norm": MAX_NORM,
               "tensors": len(full), "elements": sum(t.numel() for t in full),
               "one_gpu": {"torch_norm": ref["torch_norm"], "exact_norm": ref["exact_norm"]}, "layouts": rows}
        open(args.out, "w").write(json.dumps(rep, indent=1) + "\n"); print(json.dumps(rep))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
