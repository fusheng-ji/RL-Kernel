# SPDX-License-Identifier: Apache-2.0
"""#730 (h3-row-mean-v1) across GPU types: the bytes of ``row_sums`` / ``fixed_order_mean``.

The row sums use only FP32 loads and adds in a fixed pairwise order (no architecture-specific
instruction), so the bytes are expected to be the same on every NVIDIA GPU. This writes the
SHA-256 of each case's output; with ``--compare <reference.json>`` it also reports, per case,
whether the bytes equal the reference's (the B200 reference ships next to this script).

    CUDA_VISIBLE_DEVICES=0 python row_mean_bytes_probe.py --out h100.json \
        --compare row_mean_bytes_b200.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform

import torch

# (rows, row length): chunk and block edges, the H3 audio latent and the H3 video latent
CASES = [(3, 1), (3, 1023), (3, 1024), (3, 1025), (2, 8193), (4, 3 * 8192 + 5), (2, 2 * 136 * 32), (2, 24 * 32 * 48 * 84)]


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    args = ap.parse_args()

    from rl_engine.backends.shared.triton.reduction import row_mean
    from rl_engine.models.minimax_h3.transition import fixed_order_mean

    out: dict = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cases": {},
    }
    for rows, k in CASES:
        g = torch.Generator().manual_seed(rows * 1_000_003 + k)
        x = (torch.randn(rows, k, generator=g) * torch.logspace(-3, 3, rows)[:, None]).float().cuda()
        out["cases"][f"row_sums_{rows}x{k}"] = _digest(row_mean.row_sums(x))
        out["cases"][f"fixed_order_mean_{rows}x{k}"] = _digest(fixed_order_mean(x))
    if args.compare:
        ref = json.load(open(args.compare))
        same = {name: ref["cases"].get(name) == d for name, d in out["cases"].items()}
        out["compared_with"] = {"device": ref["device"], "capability": ref["capability"]}
        out["bytes_equal"] = same
        out["all_bytes_equal"] = all(same.values())
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "cases"}))


if __name__ == "__main__":
    main()
