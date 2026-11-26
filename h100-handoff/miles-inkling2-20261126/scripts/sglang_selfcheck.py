#!/usr/bin/env python3
# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""H4b: SGLang self-consistency on rollout 0 of H3 (single GPU, the engine flags miles uses)."""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.request

import torch

SG_TP, SG_MEM = os.environ.get("SG_TP", "1"), os.environ.get("SG_MEM", "0.55")
GPUS = os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(str(i) for i in range(int(SG_TP)))
sys.path.insert(0, "/root/miles")
ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--deterministic", action="store_true")
ap.add_argument("--log", required=True)
a = ap.parse_args()
MODEL = os.environ["WORK"] + "/models/Inkling-2-Mini-bf16"
PORT = int(os.environ.get("INK_PORT", "31000"))
samples = torch.load(f"{a.dump}/rollout_data/0.pt", weights_only=False)["samples"]


def get(s, k):
    return s[k] if isinstance(s, dict) else getattr(s, k)


items = []
for s in samples:
    toks = list(get(s, "tokens"))
    rl = int(get(s, "response_length"))
    rlp = get(s, "rollout_log_probs")
    rlp = list(rlp) if rlp is not None else None
    items.append({"tokens": toks, "rl": rl, "rlp": rlp})
# one sample per distinct prompt (each prompt was sampled 8 times); rollout 0 uses the initial weights
seen, uniq = set(), []
for it in items:
    k = tuple(it["tokens"][: len(it["tokens"]) - it["rl"]])
    if it["rlp"] is None or it["rl"] <= 8 or k in seen:
        continue
    seen.add(k)
    uniq.append(it)
items = uniq[: a.n]
cmd = [
    sys.executable,
    "-m",
    "sglang.launch_server",
    "--trust-remote-code",
    "--model-path",
    MODEL,
    "--host",
    "127.0.0.1",
    "--port",
    str(PORT),
    "--random-seed",
    "1234",
    "--skip-server-warmup",
    "--mem-fraction-static",
    SG_MEM,
    "--tp-size",
    SG_TP,
    "--page-size",
    "128",
    "--disable-radix-cache",
    "--attention-backend",
    "fa4",
    "--bf16-gemm-backend",
    "torch",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--device",
    "cuda",
    "--enable-memory-saver",
    "--no-enable-decoder-swa-bounded-replay",
    "--enable-metrics",
    "--decode-log-interval",
    "1000",
    "--lora-use-virtual-experts",
    "--enable-draft-weights-cpu-backup",
]
if a.deterministic:
    cmd.append("--enable-deterministic-inference")
log = open(a.log, "w")
p = subprocess.Popen(
    cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": GPUS}
)


def post(path, obj, timeout=600):
    r = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(r, timeout=timeout))


t0 = time.time()
while True:
    if p.poll() is not None:
        raise SystemExit("server died")
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
        break
    except Exception:
        if time.time() - t0 > 1200:
            raise SystemExit("server start timeout")
        time.sleep(5)


def score(it):
    """prefill rescoring of the dumped tokens; returns FP32 logprobs of the response tokens."""
    pl = len(it["tokens"]) - it["rl"]
    out = post(
        "/generate",
        {
            "input_ids": it["tokens"],
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": pl - 1,
        },
    )
    lp = [x[0] for x in out["meta_info"]["input_token_logprobs"]]
    # first value is None; the last rl values belong to response tokens
    return lp[-it["rl"] :]


def f32(v):
    return torch.tensor(v, dtype=torch.float32)


def cmp(x, y):
    x, y = f32(x), f32(y)
    d = (x - y).abs()
    return {
        "tokens": int(x.numel()),
        "bit_mismatch": int((x.view(torch.int32) != y.view(torch.int32)).sum()),
        "ne": int((x != y).sum()),
        "mean_abs": float(d.mean()),
        "max_abs": float(d.max()),
    }


def agg(pairs):
    r = [cmp(x, y) for x, y in pairs]
    n = sum(q["tokens"] for q in r)
    return {
        "tokens": n,
        "bit_mismatch": sum(q["bit_mismatch"] for q in r),
        "ne": sum(q["ne"] for q in r),
        "mean_abs": sum(q["mean_abs"] * q["tokens"] for q in r) / n,
        "max_abs": max(q["max_abs"] for q in r),
    }


res = {"n_samples": len(items), "deterministic": a.deterministic}
solo = [score(i) for i in items]
torch.save(
    [
        {"tokens": i["tokens"], "rl": i["rl"], "rlp": i["rlp"][-i["rl"] :], "prefill": s_}
        for i, s_ in zip(items, solo)
    ],
    a.out.replace(".json", "_scores.pt"),
)
if os.environ.get("SCORES_ONLY"):
    p.terminate()
    p.wait(120)
    time.sleep(15)
    json.dump({"n_samples": len(items)}, open(a.out, "w"))
    sys.exit(0)
solo2 = [score(i) for i in items]
with cf.ThreadPoolExecutor(16) as ex:
    conc = list(ex.map(score, items))
res["prefill_repeat_solo"] = agg(list(zip(solo, solo2)))
res["prefill_solo_vs_concurrent16"] = agg(list(zip(solo, conc)))
res["decode_rollout_vs_prefill_solo"] = agg(
    [(it["rlp"][-it["rl"] :], s) for it, s in zip(items, solo)]
)


# greedy generation: same prompt repeated solo, then inside a batch of 8 others
def gen(it, n=256):  # 16 distinct prompts
    pl = len(it["tokens"]) - it["rl"]
    o = post(
        "/generate",
        {
            "input_ids": it["tokens"][:pl],
            "sampling_params": {
                "max_new_tokens": n,
                "temperature": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
        },
    )
    return [t for _, t, _ in o["meta_info"]["output_token_logprobs"]], [
        l for l, _, _ in o["meta_info"]["output_token_logprobs"]
    ]


g1 = [gen(i) for i in items[:16]]
g2 = [gen(i) for i in items[:16]]
with cf.ThreadPoolExecutor(8) as ex:
    g3 = list(ex.map(gen, items[:16]))


def gcmp(u, v):
    tm = [sum(x == y for x, y in zip(a_[0], b_[0])) / max(len(a_[0]), 1) for a_, b_ in zip(u, v)]
    same = sum(a_[0] == b_[0] for a_, b_ in zip(u, v))
    lp = [(a_[1], b_[1]) for a_, b_ in zip(u, v) if a_[0] == b_[0]]
    first = [
        next((i for i, (x, y) in enumerate(zip(a_[0], b_[0])) if x != y), None)
        for a_, b_ in zip(u, v)
    ]
    return {
        "prompts": len(u),
        "same_tokens": same,
        "first_divergence": first,
        "mean_token_match": sum(tm) / len(tm),
        "logprob_on_same": agg(lp) if lp else None,
    }


res["greedy_repeat_solo"] = gcmp(g1, g2)
res["greedy_solo_vs_concurrent8"] = gcmp(g1, g3)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
p.terminate()
try:
    p.wait(60)
except Exception:
    p.kill()
