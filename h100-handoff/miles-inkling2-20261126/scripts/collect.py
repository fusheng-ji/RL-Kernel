#!/usr/bin/env python3
# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""Turn a native Miles run directory into the RL-Kernel Inkling report fields.

Two-node H100 version: memory and SGLang metrics come from one file per node (gpu_mem_node*.csv, sglang_metrics_node*.jsonl
in the stage directory), and latency deltas are summed over the nodes.
usage: collect.py <run dir with train.log> --stage <stage dir> --name <name> --out <report dir> --h1 <H1 run dir> [--dump <dump dir>]
Outputs <out>/<name>_perf.json (performance.py fields) and, when a dump exists, <name>_e2e.json (e2e_acceptance.py fields).
Definitions (also written to <name>_notes.json):
  prefill_ms / decode_ms: mean SGLang time-to-first-token / inter-token latency in the step, from cumulative
    Prometheus sums and counts scraped every 20 s (TTFT includes queueing inside the engine, so it is not pure prefill).
  peak_memory_bytes: max over GPUs and 5 s nvidia-smi samples in the step window (whole GPU, all processes).
  step window: from the end of the previous step to the end of this one.
  generated_tokens: sum of response lengths of the 256 samples.
  bit_mismatches: numerically different FP32 tokens (rollout vs trainer no-grad scoring); +0.0/-0.0 counted equal.
"""
import argparse
import collections
import datetime
import glob
import hashlib
import json
import os
import re
import sys

W = os.environ["WORK"]
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stage", required=True)
ap.add_argument("--h1", required=True)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--dump")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


log = open(a.run + "/train.log", errors="replace").read()
ts = (
    lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    .replace(tzinfo=datetime.timezone.utc)
    .timestamp()
)
rows, end = collections.defaultdict(dict), {}
for m in re.finditer(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) [^\]]*\][^\n]*?(step|rollout|perf) (\d+): (\{[^\n]*\})",
    log,
):
    t, kind, n, body = m.groups()
    n = int(n)
    body = re.sub(r"\b(nan|inf)\b", "None", body.rstrip())
    import ast

    try:
        rows[n].update(ast.literal_eval(body))
    except (ValueError, SyntaxError):
        continue
    if kind == "perf" and "perf/step_time" in body:
        end[n] = ts(t)
steps = sorted(end)
n = len(steps)
print("steps", n, "missing keys", [s for s in steps if "train/loss" not in rows[s]])
# windows
start = {steps[0]: end[steps[0]] - rows[steps[0]]["perf/step_time"]}
for p, s in zip(steps, steps[1:]):
    start[s] = end[p]
# memory
mem = []
for l in (x for f in sorted(glob.glob(a.stage + "/gpu_mem_node*.csv")) for x in open(f)):
    p = l.split(", ")
    if len(p) < 3 or "MiB" not in p[2]:
        continue
    t = (
        datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f")
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )
    mem.append((t, int(p[2].split()[0]) * 1048576))
# scraped latency (cumulative per node, summed over that node's engines); engines per node = 4 / SG_TP
cfg = dict(l.split("=") for l in open(a.run + "/h100_config.txt").read().split())
sg_tp = int(cfg["SG_TP"])
per_node = 4 // sg_tp
nodes = []
for mp in sorted(glob.glob(a.stage + "/sglang_metrics_node*.jsonl")):
    samp = []
    for l in open(mp):
        d = json.loads(l)
        agg = collections.Counter()
        ok = 0
        for k, v in d.items():
            if isinstance(v, dict) and "time_to_first_token_seconds_sum" in v:
                ok += 1
                agg.update(v)
        if ok == per_node:
            samp.append((d["t"], agg))
    nodes.append(samp)


def lat(s):
    S, C = collections.Counter(), collections.Counter()
    for samp in nodes:
        x = [q for q in samp if start[s] - 5 <= q[0] <= end[s] + 25]
        pre = [q for q in samp if q[0] <= start[s] + 1]
        if not x or not pre:
            return None, None
        f, l = pre[-1][1], x[-1][1]
        for k in ("time_to_first_token_seconds", "inter_token_latency_seconds"):
            S[k] += l[k + "_sum"] - f[k + "_sum"]
            C[k] += l[k + "_count"] - f[k + "_count"]
    if len(nodes) != 2:
        return None, None
    d = lambda k: S[k]
    c = lambda k: C[k]
    k1, k2 = "time_to_first_token_seconds", "inter_token_latency_seconds"
    return (1e3 * d(k1) / c(k1) if c(k1) > 0 else None, 1e3 * d(k2) / c(k2) if c(k2) > 0 else None)


meta = dict(
    mode="native",
    capture_mode="real-weight",
    platform="cuda",
    hardware="H100",
    gpu_count=8,
    nodes=2,
    precision="bf16-dequant",
    ep=8,
    rollout_tp=sg_tp,
    sglang_mem_fraction=float(cfg["SG_MEM"]),
    steps=n,
    warmup_steps=a.warmup,
    miles_revision="1e1d3fe8",
    sglang_revision="99f0be56+partial-load-patch",
    checkpoint_sha256=hashlib.sha256(
        json.dumps(json.load(open(a.h1 + "/bf16_tensors.json")), sort_keys=True).encode()
    ).hexdigest(),
    dataset_sha256=sha(W + "/datasets/dapo-math-17k/dapo-math-17k.jsonl"),
    recipe_sha256=sha("/root/miles/scripts/run_inkling2_mini.py"),
)
samples = []
for s in steps[a.warmup :]:
    gt = int(round(rows[s]["rollout/response_lengths"] * 256))
    pm, dm = lat(s)
    pk = max((v for t, v in mem if start[s] <= t <= end[s]), default=None)
    samples.append(
        {
            "step": s,
            "generated_tokens": gt,
            "step_seconds": rows[s]["perf/step_time"],
            "prefill_ms": pm,
            "decode_ms": dm,
            "peak_memory_bytes": pk,
            **{
                k: rows[s].get(k)
                for k in (
                    "perf/rollout_time",
                    "perf/train_time",
                    "perf/actor_train_time",
                    "perf/log_probs_time",
                    "perf/update_weights_time",
                    "perf/tokens_per_gpu_per_sec",
                    "rollout/truncated_ratio",
                )
            },
        }
    )
miss = [
    x["step"] for x in samples if None in (x["prefill_ms"], x["decode_ms"], x["peak_memory_bytes"])
]
json.dump(
    {**meta, "samples": samples, "missing_latency_or_memory_steps": miss},
    open(f"{a.out}/{a.name}_perf.json", "w"),
    indent=1,
)
curve = [
    {
        "step": s,
        "reward": rows[s].get("rollout/raw_reward"),
        "kl": rows[s].get("train/train_rollout_kl"),
        "abs_diff": rows[s].get("train/train_rollout_logprob_abs_diff"),
        "loss": rows[s].get("train/loss"),
        "grad_norm": rows[s].get("train/grad_norm"),
        "truncated": rows[s].get("rollout/truncated_ratio"),
        "step_time": rows[s]["perf/step_time"],
    }
    for s in steps
]
json.dump(curve, open(f"{a.out}/{a.name}_curve.json", "w"), indent=1)
if a.dump:
    import torch

    cps = 32
    by = collections.defaultdict(list)
    for f in glob.glob(a.dump + "/policy_loss_debug/rank_*_call_*.pt"):
        r, c = map(int, re.search(r"rank_(\d+)_call_(\d+)", f).groups())
        by[c // cps].append(f)
    sm = []
    for i, s in enumerate(steps):
        tok = ne = bit = 0
        for f in by.get(i, []):
            for smp in torch.load(f, weights_only=False, map_location="cpu")["samples"]:
                m = smp["local_loss_mask"].bool().flatten()
                o = smp["old_log_probs"].flatten()[m].float()
                r0 = smp["rollout_log_probs"].flatten()[m].float()
                tok += len(o)
                ne += int((o != r0).sum())
                bit += int((o.view(torch.int32) != r0.view(torch.int32)).sum())
        rp = a.dump + f"/rollout_data/{s}.pt"
        sm.append(
            {
                "step": i + 1,
                "selected_tokens": tok,
                "bit_mismatches": ne,
                "raw_bit_mismatches_incl_signed_zero": bit,
                "replay_sha256": sha(rp) if os.path.exists(rp) else None,
                "loss": rows[s]["train/loss"],
                "reward": rows[s]["rollout/raw_reward"],
                "kl": rows[s]["train/train_rollout_kl"],
                "files": len(by.get(i, [])),
            }
        )
    json.dump({**meta, "step_metrics": sm}, open(f"{a.out}/{a.name}_e2e.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
print("perf steps", len(samples), "without latency/memory:", miss)
