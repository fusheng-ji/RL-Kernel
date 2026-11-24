# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""One-GPU check of Transformer Engine fused attention (cuDNN), forward and backward.

Fails with "Multiple libcudart libraries found" when the host exposes both
libcudart.so.12 and libcudart.so.13 to an unpatched TE 2.11 build.
"""

import os

os.environ["NVTE_FLASH_ATTN"] = "0"  # force the cuDNN path

import torch  # noqa: E402
import transformer_engine.pytorch as te  # noqa: E402


def main() -> None:
    attention = te.DotProductAttention(
        32,
        128,
        num_gqa_groups=8,
        attention_dropout=0.0,
        qkv_format="sbhd",
        attn_mask_type="causal",
    ).cuda()
    query, key, value = (
        torch.randn(256, 1, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for heads in (32, 8, 8)
    )
    output = attention(query, key, value)
    output.float().sum().backward()
    print("OK", tuple(output.shape), torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
