import torch
from rl_engine.models.qwen3_next.qwen3_next_forward import shared_linear
w = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
for bias_grad in (True, False):
    b = torch.randn(64, device="cuda", dtype=torch.bfloat16, requires_grad=bias_grad)
    x = torch.randn(0, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w.grad = None
    shared_linear(x, w, b).sum().backward()
    assert x.grad.shape == x.shape and torch.count_nonzero(w.grad) == 0
    if bias_grad:
        assert b.grad is not None and torch.count_nonzero(b.grad) == 0
    else:
        assert b.grad is None
    print("bias requires_grad", bias_grad, "-> bias.grad", None if b.grad is None else tuple(b.grad.shape), "ok")
