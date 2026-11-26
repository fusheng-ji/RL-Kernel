#!/usr/bin/env python3
# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""H7: do the live engines hold exactly the weights a fresh server loads from the matching HF export?
usage: sync_compare.py <H6 stage dir> <H7a run dir> <export>:<fresh name> ...
live_node*.json (sidecar, export index = order of 'saved HF checkpoint' lines) vs <fresh name>_checksum.json (score_rollout.py).
Exports are counted from 0, so 0 = after rollout 32, 1 = 65, 2 = 98, 3 = 99. Both sides must use the same SG_TP (same shards)."""
import glob
import json
import sys

tp = lambda r, i: (r.get("parallelism_info") or [{}])[0].get("tp_rank", i)
st, fr = sys.argv[1], sys.argv[2]
live = [e for f in sorted(glob.glob(st + "/live_node*.json")) for e in json.load(open(f))]
res = []
for spec in sys.argv[3:]:
    k, name = spec.split(":")
    k = int(k)
    f = json.load(open(f"{fr}/{name}_checksum.json"))
    ft = {(tp(r, i), n): h for i, r in enumerate(f["ranks"]) for n, h in r["checksums"].items()}
    engines = []
    for e in live:
        if e["export"] != k:
            continue
        lt = {
            (tp(r, i), n): h
            for i, r in enumerate(e.get("ranks", []))
            for n, h in r["checksums"].items()
        }
        diff = [n for n in ft if lt.get(n) != ft[n]] + [n for n in lt if n not in ft]
        engines.append(
            {
                "port": e["port"],
                "same_engine_checksum": e.get("per_engine_checksum")
                == f.get("per_engine_checksum"),
                "error": e.get("error"),
                "live_tensors": len(lt),
                "fresh_tensors": len(ft),
                "differ": len(diff),
                "first": [list(x) for x in diff[:3]],
            }
        )
    res.append(
        {
            "export": k,
            "fresh": name,
            "engines": engines,
            "all_equal": bool(engines)
            and all(x["differ"] == 0 and not x["error"] for x in engines),
        }
    )
print(json.dumps(res, indent=1))
