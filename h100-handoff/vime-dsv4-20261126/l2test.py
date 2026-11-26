import torch
from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm as te
import amp_C
from apex.multi_tensor_apply import multi_tensor_applier as ax
def t(name, f, grads):
    try:
        b = torch.zeros(1, dtype=torch.int, device='cuda')
        n, _ = f[0](f[1], b, [grads], False)
        print(name, 'norm=', n.item(), 'overflow=', b.item(), flush=True)
    except Exception as e: print(name, 'EXC', repr(e)[:150], flush=True)
TE=(multi_tensor_applier, te); AX=(ax, amp_C.multi_tensor_l2norm)
def cases(dt):
    return {
     'plain': [torch.randn(1024,128,device='cuda',dtype=dt)*1e-3],
     'many+zero': [torch.randn(4096,device='cuda',dtype=dt)*1e-3, torch.zeros(4096,device='cuda',dtype=dt)],
     'empty': [torch.randn(64,device='cuda',dtype=dt), torch.empty(0,device='cuda',dtype=dt)],
     'tiny': [torch.full((256,),1e-30,device='cuda',dtype=dt)],
     'unaligned': [torch.randn(1000,device='cuda',dtype=dt)[1:777]],
     'big': [torch.randn(30_000_000,device='cuda',dtype=dt)*1e-3],
    }
for dt in (torch.bfloat16, torch.float32):
    for k, g in cases(dt).items():
        ref = sum((x.float()**2).sum().item() for x in g)**.5
        print(str(dt), k, 'ref=', ref); t('  TE  ', TE, g); t('  apex', AX, g)
g = [torch.randn(4096,device='cuda',dtype=torch.bfloat16)*1e-3 for _ in range(4)] + [torch.randn(4096,device='cuda',dtype=torch.float32)*1e-3 for _ in range(4)]
ref = sum((x.float()**2).sum().item() for x in g)**.5
for nm, lst in (('mixed bf16-first', g), ('mixed fp32-first', g[::-1])):
    print(nm, 'ref=', ref); t('  TE  ', TE, lst); t('  apex', AX, lst)
for dt in (torch.bfloat16, torch.float32):
    sub = [x for x in g if x.dtype == dt]
    print('grouped', dt, 'ref=', sum((x.float()**2).sum().item() for x in sub)**.5); t('  TE  ', TE, sub)
