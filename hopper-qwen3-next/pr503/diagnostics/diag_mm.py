import torch, os
from rl_engine.models.qwen3_next import qwen3_next_forward as p
lin=p._vllm_linear()
torch.manual_seed(0)
for (K,N,name) in [(2048,512,'dactivated: grad[n,2048]@down.T? '),(512,2048,'x'),(1024,2048,'dx: dproj[n,1024]@gate_up.T')]:
    W=(torch.randn(N,K,device='cuda')*0.02).bfloat16()
    X=torch.randn(1024,K,device='cuda').bfloat16()
    alone=lin(X[:8],W)
    res={}
    for m in (8,9,16,17,33,64,100,256,1024):
        for pos in ('first','last'):
            if pos=='first': y=lin(X[:m],W)[:8]
            else:
                Y=X[:m].clone(); Y[m-8:]=X[:8]; y=lin(Y,W)[m-8:]
            res[(m,pos)]=bool(torch.equal(y,alone)) if m>=8 else None
    print(name,K,N,{f'{k[0]}{k[1][0]}':v for k,v in res.items() if not v})
print('--- small M vs row0')
for (K,N) in [(2048,512),(512,2048),(1024,2048)]:
    W=(torch.randn(N,K,device='cuda')*0.02).bfloat16(); X=torch.randn(256,K,device='cuda').bfloat16()
    r1=lin(X[:1],W)
    print(K,N,{m:bool(torch.equal(lin(X[:m],W)[:1],r1)) for m in (1,2,3,4,5,8,16,32,64,128,256)})
