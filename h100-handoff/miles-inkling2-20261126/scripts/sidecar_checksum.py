#!/usr/bin/env python3
# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""H6 sidecar, one per node. After each HF export of the trainer, checksum every live engine of this job (during the next rollout)."""
import json
import os
import re
import sys
import time
import urllib.request

LOG, OUT, N = sys.argv[1], sys.argv[2], int(sys.argv[3])  # N: max exports to follow


def engines():
    out = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
        except OSError:
            continue
        if "sglang.launch_server" not in cmd or "/bin/sh" in cmd:
            continue
        m = {k: re.search(rf"--{k} (\S+)", cmd) for k in ("host", "port", "admin-api-key")}
        if m["host"] and m["port"] and m["admin-api-key"]:
            out[m["port"].group(1)] = (m["host"].group(1), m["admin-api-key"].group(1))
    return out


def checksum(host, port, key):
    r = urllib.request.Request(
        f"http://{host}:{port}/weights_checker",
        data=json.dumps({"action": "checksum", "skip_tensor_list": ["mlp.gate._one"]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    return json.load(urllib.request.urlopen(r, timeout=300))


seen, res = 0, []
t0 = time.time()
while seen < N and time.time() - t0 < 24 * 3600:
    time.sleep(5)
    try:
        n = sum("saved HF checkpoint to" in l for l in open(LOG, errors="replace"))
    except OSError:
        continue
    if n <= seen:
        continue
    seen = n
    time.sleep(50)  # offload + weight sync take ~20 s; a rollout takes > 90 s
    for port, (host, key) in sorted(engines().items()):
        try:
            c = checksum(host, port, key)
            c["ranks"] = [{k: v for k, v in r.items()} for r in c.get("ranks", [])]
        except Exception as e:
            c = {"error": repr(e)[:300]}
        c.update(export=seen - 1, port=port)
        c.pop("message", None)
        res.append(c)
    json.dump(res, open(OUT, "w"))
