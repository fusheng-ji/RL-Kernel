#!/usr/bin/env python3
"""Pick driver taskset CPUs disjoint from actor PIDs in a placement snap. usage: areal_pick_driver_cpus.py PLACEMENT.txt ALLOWED_CSV"""
import re, sys

placement, allowed_s = sys.argv[1], sys.argv[2]
allowed = set()
for part in allowed_s.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        a, b = part.split("-", 1)
        allowed.update(range(int(a), int(b) + 1))
    else:
        allowed.add(int(part))
actor = set()
pid = None
for line in open(placement, errors="replace"):
    if re.search(r"run_trainer|Megatron|ray::.*actor", line, re.I):
        m = re.search(r"pid=(\d+)", line)
        if m:
            pid = m.group(1)
    elif pid and line.startswith("Cpus_allowed_list:"):
        lst = line.split(":", 1)[1].strip()
        for p in lst.split(","):
            p = p.strip()
            if "-" in p:
                a, b = p.split("-", 1)
                actor.update(range(int(a), int(b) + 1))
            else:
                actor.add(int(p))
        pid = None
free = sorted(allowed - actor)
if len(free) < 8:
    free = sorted(allowed)
print(",".join(map(str, free[:32])))