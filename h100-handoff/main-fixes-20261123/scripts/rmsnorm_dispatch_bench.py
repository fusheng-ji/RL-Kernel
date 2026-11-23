"""#747 evidence: what kernel_registry.get_op("rms_norm") returns, and how the
native op (main) and the CUDA op (this branch) compare.

    PYTHONPATH=<branch worktree> python rmsnorm_dispatch_bench.py <out.json> [<main worktree>]
"""
import json
import os
import subprocess
import sys

import torch

from rl_engine.kernels.ops.cuda.norm.rmsnorm import RMSNormCudaOp
from rl_engine.kernels.ops.pytorch.norm.rms_norm import NativeRMSNormOp

EPS = 1e-6
SHAPES = [(1, 4096), (16, 4096), (256, 4096), (4096, 4096), (16384, 4096), (4096, 8192), (16384, 8192)]


def registry_choice(tree):
    code = "from rl_engine.runtime.registry import kernel_registry as r; print(type(r.get_op('rms_norm')).__name__)"
    env = {**os.environ, "PYTHONPATH": tree}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True, cwd=tree)
    return out.stdout.strip().splitlines()[-1]


def timed(fn, iters=100, repeats=5):
    for _ in range(10):
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
    out_path = sys.argv[1]
    main_tree = sys.argv[2] if len(sys.argv) > 2 else None
    import rl_engine

    branch_tree = rl_engine.__file__.rsplit("/rl_engine/", 1)[0]
    native, cuda = NativeRMSNormOp(), RMSNormCudaOp()
    rows = []
    for tokens, hidden in SHAPES:
        g = torch.Generator(device="cuda").manual_seed(tokens * 7 + hidden)
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=g)
        w = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=g)).to(torch.bfloat16)
        dy = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=g)
        x64, w64 = x.double(), w.double()
        ref = x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + EPS) * w64
        row = {"tokens": tokens, "hidden": hidden}
        outs = {}
        for name, op in (("native", native), ("cuda", cuda)):
            y = op.forward(x, w, eps=EPS)
            outs[name] = y
            row[f"{name}_max_abs_err_vs_fp64"] = float((y.double() - ref).abs().max())
            row[f"{name}_run_to_run_bitwise"] = bool(torch.equal(y, op.forward(x, w, eps=EPS)))
            # Batch invariance: the first row alone, and the batch split in two, against the full batch.
            alone = op.forward(x[:1].clone(), w, eps=EPS)
            half = tokens // 2
            split = torch.cat([op.forward(x[:half].clone(), w, eps=EPS), op.forward(x[half:].clone(), w, eps=EPS)]) if half else y
            row[f"{name}_batch_invariant"] = bool(torch.equal(alone[0], y[0]) and torch.equal(split, y))
            row[f"{name}_fwd_ms"] = timed(lambda: op.forward(x, w, eps=EPS))
            xg, wg = x.clone().requires_grad_(), w.clone().requires_grad_()

            def fwd_bwd():
                xg.grad = wg.grad = None
                op.forward(xg, wg, eps=EPS).backward(dy)

            row[f"{name}_fwd_bwd_ms"] = timed(fwd_bwd, iters=50)
        row["outputs_bitwise_equal"] = bool(torch.equal(outs["native"], outs["cuda"]))
        row["max_abs_diff_native_vs_cuda"] = float((outs["native"].float() - outs["cuda"].float()).abs().max())
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": "bfloat16",
        "eps": EPS,
        "registry_choice": {"branch": registry_choice(branch_tree)},
        "rows": rows,
    }
    if main_tree:
        result["registry_choice"]["main"] = registry_choice(main_tree)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
