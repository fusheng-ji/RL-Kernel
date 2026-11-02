import sys, time, torch
sys.path.insert(0, ".")
from rl_engine.backends.extension import _C
import rl_engine
from rl_engine.backends.cuda.model_specific.minimax_h3 import adaln_row_gather as arg
from rl_engine.backends.cuda.model_specific.minimax_h3.gate_residual import H3GateResidualCudaOp
from rl_engine.backends.cuda.model_specific.minimax_h3.rmsnorm import H3RMSNormCudaOp

print(rl_engine.__file__, _C.__file__)
dev = "cuda"
S, H, T = 4097, 5376, 3
g = torch.Generator(device=dev).manual_seed(0)
tag = torch.randint(0, 3, (S,), device=dev, generator=g)
ts = torch.randint(0, T, (S,), device=dev, generator=g)
index = (ts * 3 + tag).contiguous()
rows = T * 3

def timeit(fn, n=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

print(f"_segment_tiles           {timeit(lambda: arg._segment_tiles(index, rows)):.3f} ms")
res = torch.randn(1, S, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
y = torch.randn(1, S, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
gate = torch.randn(rows, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
gop = H3GateResidualCudaOp()
def gate_fb():
    out = gop(res, y, gate, index)
    out.backward(torch.ones_like(out))
print(f"gate_residual fwd+bwd    {timeit(gate_fb):.3f} ms")
nop = H3RMSNormCudaOp()
x = torch.randn(1, S, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(H, device=dev, dtype=torch.bfloat16, requires_grad=True)
sh = torch.randn(rows, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
sc = torch.randn(rows, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
def norm_fb():
    out = nop.forward_modulated(x, w, sh, sc, index)
    out.backward(torch.ones_like(out))
print(f"rmsnorm_mod fwd+bwd      {timeit(norm_fb):.3f} ms")
