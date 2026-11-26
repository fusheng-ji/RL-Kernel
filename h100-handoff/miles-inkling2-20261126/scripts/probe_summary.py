#!/usr/bin/env python3
# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""H2: one row per probe directory: exit code, peak memory per node, the first out-of-memory or error line and the phase it was in.
usage: probe_summary.py <probe stage dir> ... > probe_summary.json"""
import glob
import json
import os
import re
import sys

PHASE = [
    ("rollout", re.compile(r"sglang|rollout|generate", re.I)),
    ("train", re.compile(r"actor|train|backward|optimizer|fsdp", re.I)),
    ("weight sync", re.compile(r"update_weight|weights?_sync|broadcast", re.I)),
]
out = []
for st in sys.argv[1:]:
    for log in sorted(glob.glob(st + "/*/train.log")):
        txt = open(log, errors="replace").read()
        rc = re.findall(r"^train rc=(\d+)", txt, re.M)
        err = next(
            (
                l
                for l in txt.splitlines()
                if re.search(
                    r"OutOfMemory|out of memory|CUDA error|Traceback|RuntimeError|ValueError", l
                )
            ),
            None,
        )
        peak = {}
        for f in glob.glob(st + "/gpu_mem_node*.csv"):
            v = [
                int(p.split(", ")[2].split()[0])
                for p in open(f)
                if p.count(", ") >= 2 and "MiB" in p
            ]
            peak[os.path.basename(f)] = max(v) if v else None
        cfg = (
            open(os.path.dirname(log) + "/h100_config.txt").read().strip()
            if os.path.exists(os.path.dirname(log) + "/h100_config.txt")
            else None
        )
        out.append(
            {
                "probe": os.path.basename(os.path.dirname(log)),
                "config": cfg,
                "train_rc": int(rc[-1]) if rc else None,
                "steps_done": len(re.findall(r"perf \d+: \{", txt)),
                "peak_mib_per_node": peak,
                "first_error": err[:500] if err else None,
                "phase_guess": next((p for p, r in PHASE if err and r.search(err)), None),
            }
        )
print(json.dumps(out, indent=1))
