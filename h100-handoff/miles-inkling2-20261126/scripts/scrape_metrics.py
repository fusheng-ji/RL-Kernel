#!/usr/bin/env python3
# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""Every 20 s, append cumulative SGLang latency sums/counts of every live engine (GET /metrics, no auth header)."""
import json
import os
import re
import sys
import time
import urllib.request

OUT = open(sys.argv[1], "a")
PAT = re.compile(
    r"^sglang:(time_to_first_token_seconds|inter_token_latency_seconds|e2e_request_latency_seconds|prompt_tokens_total|generation_tokens_total|queue_time_seconds)(_sum|_count)?\{[^}]*\} (\S+)",
    re.M,
)


def ports():
    out = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
        except OSError:
            continue
        if "sglang.launch_server" in cmd and "/bin/sh" not in cmd:
            h, p = re.search(r"--host (\S+)", cmd), re.search(r"--port (\d+)", cmd)
            if h and p:
                out[p.group(1)] = h.group(1)
    return out


while True:
    row = {"t": time.time()}
    for port, host in ports().items():
        try:
            txt = urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=5).read().decode()
        except Exception as e:
            row[port] = {"error": type(e).__name__}
            continue
        agg = {}
        for n, s, v in PAT.findall(txt):
            agg[n + s] = agg.get(n + s, 0.0) + float(v)
        row[port] = agg
    OUT.write(json.dumps(row) + "\n")
    OUT.flush()
    time.sleep(20)
