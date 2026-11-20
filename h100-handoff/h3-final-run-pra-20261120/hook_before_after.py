"""The Megatron get_grad_norm_fp32 hook of test-h3 against this PR's, on the same gradients.

Loads rl_engine/ops/norm/reproducible.py from both checkouts (paths given), installs each into a
stand-in megatron.core.optimizer, initialises a one-rank NCCL group (the old hook needs one) and
compares the returned norms for several gradient lists. One GPU.

    python hook_before_after.py <test-h3 reproducible.py> <PR reproducible.py> <out.json>
"""
import importlib.util, json, os, sys, types
import torch
import torch.distributed as dist


def stub():
    def original(g, n=2, grp=None):
        raise AssertionError("fallback not expected for the L2 norm")
    cg = types.ModuleType("megatron.core.optimizer.clip_grads"); cg.get_grad_norm_fp32 = original
    op = types.ModuleType("megatron.core.optimizer.optimizer"); op.get_grad_norm_fp32 = original
    pk = types.ModuleType("megatron.core.optimizer"); pk.clip_grads, pk.optimizer = cg, op
    sys.modules.update({"megatron": types.ModuleType("megatron"), "megatron.core": types.ModuleType("megatron.core"),
                        "megatron.core.optimizer": pk, "megatron.core.optimizer.clip_grads": cg, "megatron.core.optimizer.optimizer": op})
    return cg


def hook(path, name):
    cg = stub()
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.install(); return cg.get_grad_norm_fp32


def main():
    base_path, head_path, out = sys.argv[1:4]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29731")
    dist.init_process_group("nccl", rank=0, world_size=1); torch.cuda.set_device(0)
    old, new = hook(base_path, "repro_base"), hook(head_path, "repro_head")
    g = torch.Generator().manual_seed(3); rows = []
    cases = {"lora_700": [(torch.randn(*s, generator=g) * 1e-3) for s in [(32, 5376), (5376, 32)] * 350],
             "mixed_scales": [torch.randn(1000, generator=g) * 10.0 ** k for k in range(-20, 21, 4)],
             "one_tensor": [torch.randn(1 << 20, generator=g)]}
    for name, ts in cases.items():
        ts = [t.cuda() for t in ts]
        a, b = old(ts, 2, None), new(ts, 2, None)
        rows.append({"case": name, "tensors": len(ts), "test_h3": a, "this_pr": b, "equal": a == b})
    dist.destroy_process_group()
    rep = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "cases": rows, "all_equal": all(r["equal"] for r in rows)}
    open(out, "w").write(json.dumps(rep, indent=1) + "\n"); print(json.dumps(rep))


if __name__ == "__main__":
    main()
