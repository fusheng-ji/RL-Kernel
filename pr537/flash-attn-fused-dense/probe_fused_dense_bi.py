"""Batch invariance of flash-attention's cuBLASLt `fused_dense_lib.linear_act_forward` (PR #537 prior art).

The library has no SiLU epilogue and accepts only FP16/BF16, so `SiLU + Linear` and the FP32 path cannot be run.
What can be measured is the BF16 Linear with its GELU and ReLU epilogues at the AdaLN modulation projection shape
(in 3072, out 6 * 3072). A singleton row (B = 1) is compared bytewise with the same row inside a batch of B rows.
"""
import json, sys
import torch
sys.path.insert(0, sys.argv[1])
import fused_dense_lib

IN, OUT = 3072, 6 * 3072
records = []
for dtype in (torch.bfloat16,):
    for seed in range(3):
        g = torch.Generator(device="cuda").manual_seed(seed)
        w = (torch.randn(OUT, IN, generator=g, device="cuda") * 0.02).to(dtype)
        b = (torch.randn(OUT, generator=g, device="cuda") * 0.02).to(dtype)
        for is_gelu in (True, False):
            for B in (64, 257, 2048):
                x = torch.randn(B, IN, generator=g, device="cuda").to(dtype)
                run = lambda rows: fused_dense_lib.linear_act_forward(rows, w, b, is_gelu, False, 0)[0]
                batched = run(x)
                differing = sum(not torch.equal(run(x[i:i + 1])[0], batched[i]) for i in range(B))
                records.append(dict(dtype=str(dtype), act="gelu" if is_gelu else "relu", seed=seed, B=B,
                                    rows_differing=differing))
failing = sum(1 for r in records if r["rows_differing"])
print(json.dumps(dict(torch=torch.__version__, device=torch.cuda.get_device_name(), records=records,
                      records_failing=failing, records_total=len(records)), indent=1))
