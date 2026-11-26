"""Summarize a suite directory (runs/core or runs/drift) into summary.json. CPU only.

    python3 analyze.py <SUITE_DIR> [<SUITE_DIR> ...] > summary.json

Per case: rc; per step the train/rollout logprob difference, entropy, raw reward and step time;
peak memory and host per trainer rank; non-finite gradient records; QAT moved fraction.
Host names are replaced by node-a, node-b, ... in order of appearance, so the output can be sent back.
Replays (train-*-0.pt; det-* against det-R0a, others against R0): mean / max |delta logp| over response tokens and the bitwise-equal fraction.
"""
import ast
import glob
import json
import os
import re
import sys

STEP = re.compile(r"model\.py:\d+ - step (\d+): (\{.*\})")
ROLL = re.compile(r"train_metric_utils\.py:\d+ - rollout (\d+): (\{.*\})")
PERF = re.compile(r"train_metric_utils\.py:\d+ - perf (\d+): (\{.*\})")
HOSTS = {}


def host(h):
    return HOSTS.setdefault(h, "node-" + "abcdefghijklmnop"[len(HOSTS)])


def parse_dict(s):
    s = re.sub(r"tensor\(\[([^\]]*)\][^)]*\)", r"\1", s)
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}


def case(d):
    r = {"rc": open(f"{d}/rc").read().strip() if os.path.exists(f"{d}/rc") else None}
    steps, roll, perf = {}, {}, {}
    if os.path.exists(f"{d}/train.log"):
        for line in open(f"{d}/train.log", errors="replace"):
            for pat, tgt in ((STEP, steps), (ROLL, roll), (PERF, perf)):
                m = pat.search(line)
                if m and int(m.group(1)) not in tgt:
                    tgt[int(m.group(1))] = parse_dict(m.group(2))
    rows = []
    for i in sorted(set(steps) | set(roll)):
        s, ro, p = steps.get(i, {}), roll.get(i, {}), perf.get(i, {})
        rows.append({"step": i, "logprob_diff": s.get("train/train_rollout_logprob_abs_diff"),
                     "entropy": s.get("train/entropy_loss"), "raw_reward": ro.get("rollout/raw_reward"),
                     "step_time_s": p.get("perf/step_time")})
    r["steps"] = rows
    r["n_steps"] = sum(x["logprob_diff"] is not None for x in rows)
    ranks, bad, grads, qat = {}, 0, 0, []
    for f in sorted(glob.glob(f"{d}/probe-rank*.jsonl")):
        for line in open(f):
            x = json.loads(line)
            k = str(x.get("rank"))
            if x["kind"] == "mem":
                e = ranks.setdefault(k, {"host": host(x["host"]), "peak_gib": 0.0})
                e["peak_gib"] = round(max(e["peak_gib"], x["peak_gib"]), 2)
            elif x["kind"] == "grads":
                grads += 1
                bad += x.get("nonfinite_tensors", 0) != 0
            elif x["kind"] == "qat" and k == "0":
                qat.append(round(x["elements_moved_frac"], 4))
    r.update(ranks=ranks, grad_records=grads, grad_records_nonfinite=bad)
    if qat:
        r["qat_moved_frac_rank0"] = qat
    times = [x["step_time_s"] for x in rows[2:] if x["step_time_s"]]
    if times:
        r["median_step_time_s_from_step2"] = round(sorted(times)[len(times) // 2], 2)
    return r


def replays(suite):
    import torch
    f = {os.path.basename(p)[len("train-"):-len("-0.pt")]: p for p in glob.glob(f"{suite}/train-*-0.pt")}
    if not f:
        return {}

    def tokens(p):
        out = {}
        for path in glob.glob(p.replace("-0.pt", "-0.pt*")) or [p]:
            d = torch.load(path, weights_only=False)
            for s in d["samples"]:
                m = s["loss_masks"].bool()
                out[s["sample_index"]] = (s["log_probs"][m].float(), s["rollout_log_probs"][m].float())
        return out

    t = {k: tokens(p) for k, p in f.items()}
    res = {}
    for k, v in t.items():
        lp = torch.cat([a for a, _ in v.values()]); rp = torch.cat([b for _, b in v.values()])
        res[f"{k}_train_vs_rollout"] = {"tokens": lp.numel(), "mean_abs": float((lp - rp).abs().mean())}
    for k in t:
        det = k.startswith("det-")
        base = next((b for b in (("det-R0a", "det-a") if det else ("R0",)) if b in t), None)
        if base and k != base:
            ids = sorted(set(t[k]) & set(t[base]))
            a = torch.cat([t[k][i][0] for i in ids]); b = torch.cat([t[base][i][0] for i in ids])
            dd = (a - b).abs()
            res[f"{k}_vs_{base}"] = {"tokens": a.numel(), "mean_abs": float(dd.mean()), "max_abs": float(dd.max()),
                                     "bitwise_equal_frac": float((dd == 0).float().mean())}
    return res


summary = {}
for suite in sys.argv[1:]:
    name = os.path.basename(os.path.normpath(suite))
    cases = {os.path.basename(c): case(c) for c in sorted(glob.glob(f"{os.path.normpath(suite)}/*")) if os.path.isdir(c) and os.path.exists(f"{c}/node0.log")}
    summary[name] = {"cases": cases, "replays": replays(suite)}
print(json.dumps(summary, indent=1))
