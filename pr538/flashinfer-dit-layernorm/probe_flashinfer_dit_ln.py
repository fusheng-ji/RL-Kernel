"""Batch-invariance and accuracy probe of flashinfer.fused_dit_residual_layernorm_scale_shift (PR #538 prior art).

Computes norm(x) * (1 + scale) + shift for the final-AdaLN stream (hidden 3072, BF16) with per-sample modulation
expanded to rows in the kernel's required temb.chunk(6, dim=2) layout. A singleton (B=1) sample is compared bytewise
with the same sample inside batches of B rows.
"""
import json, sys
import torch, flashinfer
from flashinfer.norm import fused_dit_residual_layernorm_scale_shift as dit_ln

H, device = 3072, "cuda"

def run(x, temb_scale, temb_shift):
    """x [B,S,H]; temb_* [B,H] per-sample modulation. Returns norm_out [B,S,H]."""
    B, S, _ = x.shape
    temb = torch.zeros(B, S, 6 * H, dtype=torch.bfloat16, device=device)
    chunks = temb.chunk(6, dim=2)
    chunks[1].copy_(temb_shift[:, None, :].expand(B, S, H))   # shift slot
    chunks[2].copy_(temb_scale[:, None, :].expand(B, S, H))   # scale slot
    _, out = dit_ln(x.contiguous(), chunks[2], chunks[1], epsilon=1e-6)
    return out

records, fails = [], 0
for seed in range(3):
    g = torch.Generator(device=device).manual_seed(seed)
    for S in (31, 256):
        for B in (2, 64, 257):
            x = torch.randn(B, S, H, generator=g, device=device).to(torch.bfloat16)
            sc = (0.1 * torch.randn(B, H, generator=g, device=device)).to(torch.bfloat16)
            sh = (0.1 * torch.randn(B, H, generator=g, device=device)).to(torch.bfloat16)
            batched = run(x, sc, sh)
            differing = 0
            for i in range(B):
                single = run(x[i:i + 1], sc[i:i + 1], sh[i:i + 1])
                if not torch.equal(single[0], batched[i]):
                    differing += 1
            fails += differing > 0
            records.append(dict(seed=seed, S=S, B=B, rows_differing=differing))
            # accuracy against FP64
            ref = torch.nn.functional.layer_norm(x.double(), (H,), eps=1e-6) * (1 + sc.double())[:, None] + sh.double()[:, None]
            records[-1]["max_abs_vs_fp64"] = float((batched.double() - ref).abs().max())
print(json.dumps(dict(flashinfer=flashinfer.__version__, torch=torch.__version__, device=torch.cuda.get_device_name(),
                      records=records, records_failing=fails, records_total=len(records)), indent=1))
