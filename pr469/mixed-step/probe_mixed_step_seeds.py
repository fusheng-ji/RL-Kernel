#!/usr/bin/env python3
"""Seed sweep of the single-decode-row case of probe_mixed_step.py: how many state and output elements of one
decode row differ between a decode-only step (packed kernel) and a mixed step, over many random fixtures.

usage: probe_mixed_step_seeds.py [num_seeds] > seeds.json   (run next to probe_mixed_step.py)
"""
import importlib.util
import json
import os
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("probe", os.path.join(here, "probe_mixed_step.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    res = {"num_seeds": n, "decode_rows": 1, "runs": []}
    for cfg, (H, HV) in probe.CONFIGS.items():
        for state_dtype in (torch.float32, torch.bfloat16):
            st, out = [], []
            for seed in range(n):
                d = probe.make(H, HV, 1, seed, state_dtype, num_blocks=3)
                a = probe.path_a(d, H, HV)
                c = probe.path_c(d, H, HV, 64)
                st.append(int((a["ssm"] != c["ssm"]).sum()))
                out.append(int((a["out"] != c["out"]).sum()))
            s = sorted(st)
            res["runs"].append({
                "config": cfg, "state_dtype": str(state_dtype).split(".")[-1],
                "state_elements": int(d["ssm"][1].numel()), "output_elements": int(a["out"].numel()),
                "state_differing": {"min": s[0], "median": s[len(s) // 2], "max": s[-1], "seeds_with_any": sum(x > 0 for x in st)},
                "output_differing": {"max": max(out), "seeds_with_any": sum(x > 0 for x in out),
                                     "seeds_with_exactly_one": sum(x == 1 for x in out),
                                     "histogram": {str(k): out.count(k) for k in sorted(set(out))}},
                "per_seed_state_differing": st, "per_seed_output_differing": out,
            })
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
