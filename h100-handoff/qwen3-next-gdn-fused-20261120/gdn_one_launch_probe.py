"""One GPU: the one-launch GDN recurrence against the per-token decode loop it replaces.

    python gdn_one_launch_probe.py <SRC> > probe.jsonl

<SRC> is a checkout of perf/qwen3-next-gdn-fused-recurrence (or a branch containing it).
For each case it prints one JSON line: whether output and final state are torch.equal,
the largest absolute output difference, and the median time of each path.
Shapes are one TP4 rank of Qwen3-Next's GDN: 4 key heads, 8 value heads, head dim 128.
"""

import json
import sys
import time

import torch

sys.path.insert(0, sys.argv[1])

from rl_engine.integrations.engines.train.vllm.qwen3_next_gdn import (  # noqa: E402
    decode_loop_sequence,
    packed_recurrent_sequence,
)

DEV = torch.device("cuda")
CASES = [((64,), 65), ((1024,), 65), ((1024, 777, 1500, 64), 65), ((512,) * 16, 362)]


def timed(function, repeats):
    function()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return sorted(values)[len(values) // 2] * 1e3


with torch.no_grad():
    for lengths, blocks in CASES:
        generator = torch.Generator(device=DEV).manual_seed(len(lengths) * 7 + lengths[0])
        tokens = sum(lengths)
        qkv = (torch.randn(tokens, 2048, device=DEV, generator=generator) * 0.3).bfloat16()
        a = torch.randn(tokens, 8, device=DEV, generator=generator).bfloat16()
        b = torch.randn(tokens, 8, device=DEV, generator=generator).bfloat16()
        A_log = torch.randn(8, device=DEV, generator=generator) * 0.5
        dt_bias = torch.randn(8, device=DEV, generator=generator) * 0.5
        bank = torch.randn(blocks, 8, 128, 128, device=DEV, generator=generator) * 0.01
        indices = torch.arange(1, len(lengths) + 1, device=DEV, dtype=torch.int32)
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
        args = (qkv, a, b, A_log, dt_bias, bank, indices, cu)
        new_out, new_state = packed_recurrent_sequence(*args, num_k_heads=4)
        old_out, old_state = decode_loop_sequence(*args, num_k_heads=4)
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "lengths": list(lengths),
                    "cache_slots": blocks,
                    "output_equal": torch.equal(new_out, old_out),
                    "state_equal": torch.equal(new_state, old_state),
                    "output_max_abs": (new_out.float() - old_out.float()).abs().max().item(),
                    "one_launch_ms": timed(
                        lambda: packed_recurrent_sequence(*args, num_k_heads=4), 5
                    ),
                    "decode_loop_ms": timed(lambda: decode_loop_sequence(*args, num_k_heads=4), 3),
                }
            ),
            flush=True,
        )
