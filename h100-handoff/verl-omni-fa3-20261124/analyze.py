"""Summaries of run_arm.sh runs (CPU).

  analyze.py RUN [RUN ...] [--repeat A B] --out summary.json

Per run: exit, peak memory, per-step timings (step 1 is warm-up, the rest is the steady part), the recipe's own log-prob
difference. With a capture: actor against rollout on the same transition, paired by the digest of the next latent, and
the attention backends / kernel files each process reported. ``--repeat A B`` compares the rollout next-latent digests
of two captured runs (step 1 of both starts from the same weights and seeds).
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics as st

import torch


def timings(run):
    rows = []
    for line in open(os.path.join(run, "steps.txt")):
        if not line.startswith("step:"):  # a progress bar's carriage return can split a line
            continue
        d = dict(re.findall(r"(\S+?):(-?[0-9.]+(?:e[+-]?[0-9]+)?)(?= |$)", line))
        rows.append({k: float(v) for k, v in d.items() if k.startswith("timing_s/") or "logprob_abs_diff" in k or k == "step"})
    steady = rows[1:]
    keys = sorted({k for r in rows for k in r if k.startswith("timing_s/")})
    out = {"steps": rows, "steady_steps": len(steady)}
    out["steady_median"] = {k: st.median(r[k] for r in steady if k in r) for k in keys if any(k in r for r in steady)}
    return out


def load(cap, side):
    out = collections.defaultdict(list)
    for p in sorted(glob.glob(f"{cap}/{side}/*.pt")):
        out[os.path.basename(p).split(".")[0]].append(torch.load(p, weights_only=True))
    return out


def capture(run):
    cap = os.path.join(run, "cap")
    if not os.path.isdir(cap):
        return None
    ro, ac = load(cap, "rollout"), load(cap, "actor")
    backends = collections.Counter()
    for p in glob.glob(f"{cap}/backends/*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            backends[f"{r['kind']}: {re.sub(r'^.*?(models--|kernels--)', r'<HUB>/', r['value'])}"] += 1
    groups = collections.defaultdict(list)
    for k in sorted(set(ro) & set(ac)):
        r = ro[k][0]
        for a in ac[k]:
            rv, av = r["velocity"].flatten(), a["velocity"].flatten()
            if rv.shape != av.shape:
                raise SystemExit(f"{k}: shapes differ {tuple(r['velocity'].shape)} {tuple(a['velocity'].shape)}")
            d = (av - rv).double()
            groups["video" if r["velocity"].shape[-1] == 96 else "audio", "grad" if a["grad"] else "nograd"].append({
                "denoise_step": r["step"], "input_equal": bool(torch.equal(a["sample"].flatten(), r["sample"].flatten())),
                "differ": int((d != 0).sum()), "rel_l2": float(d.norm() / rv.double().norm()), "max_abs": float(d.abs().max()),
                "log_prob_diff": float((a["log_prob"].flatten() - r["log_prob"].flatten()).abs().max()), "numel": rv.numel()})
    out = {"rollout_transitions": len(ro), "actor_transitions": len(ac), "matched": len(set(ro) & set(ac)),
           "rollout_tp_copies_that_differ": sum(not torch.equal(v[0]["velocity"], x["velocity"]) for v in ro.values() for x in v[1:]),
           "backends": dict(sorted(backends.items())), "groups": {}}
    for key, v in sorted(groups.items()):
        by = collections.defaultdict(list)
        for x in v:
            by[x["denoise_step"]].append(x["rel_l2"])
        out["groups"]["/".join(key)] = {
            "pairs": len(v), "input_equal": sum(x["input_equal"] for x in v), "bitwise": sum(x["differ"] == 0 for x in v),
            "numel": v[0]["numel"], "rel_l2_median_by_denoise_step": {s: st.median(q) for s, q in sorted(by.items())},
            **{f"{q}_min_med_max": [min(x[q] for x in v), st.median(x[q] for x in v), max(x[q] for x in v)]
               for q in ("rel_l2", "max_abs", "log_prob_diff")}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--repeat", nargs=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"runs": {}}
    for run in a.runs:
        name = os.path.basename(os.path.normpath(run))
        r = {"commit": open(f"{run}/commit.txt").read().strip(), "summary": open(f"{run}/summary.txt").read().split("\n"),
             "timing": timings(run), "capture": capture(run)}
        log = open(f"{run}/train.log", errors="ignore").read()
        r["rollout_backend_log_lines"] = len(re.findall(r"kernels-backed attention backend 'FLASH_ATTN_3_HUB'", log))
        res["runs"][name] = r
        print(name, r["summary"][0], json.dumps({k.split("/")[1]: round(v, 1) for k, v in r["timing"]["steady_median"].items()}))
        for g, v in (r["capture"] or {}).get("groups", {}).items():
            print("  ", g, "pairs", v["pairs"], "input equal", v["input_equal"], "bitwise", v["bitwise"],
                  "rel-L2 min/med/max %.4f %.4f %.4f" % tuple(v["rel_l2_min_med_max"]))
    if a.repeat:
        ka, kb = (set(os.path.basename(p).split(".")[0] for p in glob.glob(f"{x}/cap/rollout/*.pt")) for x in a.repeat)
        res["repeat"] = {"runs": [os.path.basename(os.path.normpath(x)) for x in a.repeat], "transitions": [len(ka), len(kb)],
                         "same_next_latent": len(ka & kb)}
        print("repeat", res["repeat"])
    json.dump(res, open(a.out, "w"), indent=1)


main()
