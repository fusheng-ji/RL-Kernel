import torch
g=torch.Generator(device='cuda').manual_seed(0)
a=torch.randn(1024,2048,device='cuda',generator=g)
ref=a[:8].sum(-1)
print({t: bool(torch.equal(a[:t].sum(-1)[:8],ref)) for t in (8,9,16,32,64,128,256,1024)})
print(torch.cuda.get_device_properties(0).multi_processor_count)
