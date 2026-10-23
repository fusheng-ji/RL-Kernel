import torch, sys
sys.path.insert(0,'.')
from benchmarks.models.qwen3_next import moe_route_candidates as m
torch.manual_seed(0)
w=m.make_weights(); x=m.make_tokens(1024,seed=1)
cand=m._rl_kernel(w)
g=torch.Generator(device="cuda").manual_seed(7)
dy=torch.randn(x.shape,device='cuda',generator=g).to(torch.bfloat16)
alone=m.cand_grads(cand,x[:8],dy[:8])
for size in (16,64):
    full=m.cand_grads(cand,x[:size],dy[:size])
    d=(full[0][:8].float()-alone[0].float())
    bad=[i for i in range(8) if not torch.equal(full[0][i],alone[0][i])]
    print(size,'dx equal',torch.equal(full[0][:8],alone[0]),'max',d.abs().max().item(),'bad rows',bad)
    again=m.cand_grads(cand,x[:size],dy[:size]); print(' repeat dx equal',torch.equal(again[0],full[0]),'dW repeat',[torch.equal(a,b) for a,b in zip(again[1:],full[1:])])
a2=m.cand_grads(cand,x[:8],dy[:8]); print('alone repeat',torch.equal(a2[0],alone[0]))
print('--- split paths')
from rl_engine.models.qwen3_next import qwen3_next_forward as p
def paths(rows,d):
    l=rows.detach().clone().requires_grad_(True)
    routes=p.stable_top10_routes(p.shared_router(l,w['router']))
    gu=cand['graph'](rows)[1][1]; dn=cand['graph'](rows)[1][2]
    out=p._RoutedExperts.apply(l,gu,dn,routes.indices)
    y=p.combine_routes(out,routes.weights)
    gl,gw=torch.autograd.grad(y,[l,routes.weights],d,retain_graph=True)
    # experts only
    l2=rows.detach().clone().requires_grad_(True)
    out2=p._RoutedExperts.apply(l2,gu,dn,routes.indices.detach())
    y2=p.combine_routes(out2,routes.weights.detach()); g2,=torch.autograd.grad(y2,[l2],d)
    return gl,g2,gw
a=paths(x[:8],dy[:8]); b=paths(x[:16],dy[:16])
print('total dx rows equal',[torch.equal(a[0][i],b[0][i]) for i in range(8)])
print('experts-only dx rows equal',[torch.equal(a[1][i],b[1][i]) for i in range(8)])
print('route-weight grad rows equal',[torch.equal(a[2][i],b[2][i]) for i in range(8)])
ra=p.stable_top10_routes(p.shared_router(x[:8],w['router'])); rb=p.stable_top10_routes(p.shared_router(x[:16],w['router']))
print('routes equal row2', torch.equal(ra.indices[2],rb.indices[2]), torch.equal(ra.weights[2],rb.weights[2]))
