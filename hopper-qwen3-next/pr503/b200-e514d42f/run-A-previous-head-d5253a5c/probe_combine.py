import torch
from rl_engine.models.qwen3_next import qwen3_next_forward as p
print("combine_routes:", type(p.combine_routes).__name__, "_CombineRoutes" in dir(p))
g = torch.Generator(device="cuda").manual_seed(0)
T = 1024
eo = torch.randn(T, 10, 2048, device="cuda", generator=g).to(torch.bfloat16)
w = torch.rand(T, 10, device="cuda", generator=g)
dy = torch.randn(T, 2048, device="cuda", generator=g).to(torch.bfloat16)
def grads(n):
    e = eo[:n].clone().requires_grad_(True); ww = w[:n].clone().requires_grad_(True)
    y = p.combine_routes(e, ww)
    ge, gw = torch.autograd.grad(y, [e, ww], dy[:n])
    return y.detach(), ge, gw
y0, ge0, gw0 = grads(8)
res = {}
for n in (8, 9, 16, 32, 64, 128, 256, 1024):
    y, ge, gw = grads(n)
    res[n] = dict(y=bool(torch.equal(y[:8], y0)), d_expert=bool(torch.equal(ge[:8], ge0)), d_route_weight=bool(torch.equal(gw[:8], gw0)))
for n, r in res.items(): print(n, r)
