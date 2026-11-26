#!/usr/bin/env python3
# flake8: noqa  (handoff helper kept as close as possible to the scripts behind the B200 evidence)
"""Fresh SGLang (SG_TP GPUs, memory fraction SG_MEM from h100.env) (the engine flags miles uses): weights checksum, then prefill-score every sample of one rollout, one request at a time.
args: --model <hf dir> --rollout <rollout_data/N.pt> --out <scores.pt> --log <server log> [--checksum <json>]"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

import torch

SG_TP, SG_MEM = os.environ.get("SG_TP", "1"), os.environ.get("SG_MEM", "0.55")
GPUS = os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(str(i) for i in range(int(SG_TP)))
ap = argparse.ArgumentParser()
for k in ("--model", "--rollout", "--out", "--log"):
    ap.add_argument(k, required=True)
ap.add_argument("--checksum")
a = ap.parse_args()
PORT, KEY = int(os.environ.get("INK_PORT", "31000")), "local-k3"
samples = torch.load(a.rollout, weights_only=False)["samples"]
cmd = [
    sys.executable,
    "-m",
    "sglang.launch_server",
    "--trust-remote-code",
    "--model-path",
    a.model,
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
    "--admin-api-key",
    KEY,
    "--device",
    "cuda",
    "--enable-memory-saver",
    "--no-enable-decoder-swa-bounded-replay",
    "--lora-use-virtual-experts",
]
p = subprocess.Popen(
    cmd,
    stdout=open(a.log, "w"),
    stderr=subprocess.STDOUT,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": GPUS},
)


def post(path, obj, timeout=600):
    r = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    return json.load(urllib.request.urlopen(r, timeout=timeout))


try:
    t0 = time.time()
    while True:
        if p.poll() is not None:
            raise SystemExit("server died")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            break
        except Exception:
            if time.time() - t0 > 1500:
                raise SystemExit("server start timeout")
            time.sleep(5)
    if a.checksum:
        json.dump(
            post("/weights_checker", {"action": "checksum", "skip_tensor_list": ["mlp.gate._one"]}),
            open(a.checksum, "w"),
        )
    out = []
    for s in samples:
        toks, rl = list(s["tokens"]), int(s["response_length"])
        if rl < 1 or s["rollout_log_probs"] is None:
            continue
        pl = len(toks) - rl
        o = post(
            "/generate",
            {
                "input_ids": toks,
                "sampling_params": {
                    "max_new_tokens": 0,
                    "temperature": 0,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": pl - 1,
            },
        )
        lp = [x[0] for x in o["meta_info"]["input_token_logprobs"]][-rl:]
        out.append({"rl": rl, "rlp": list(s["rollout_log_probs"])[-rl:], "prefill": lp})
    torch.save(out, a.out)
    print("scored", len(out))
finally:
    p.terminate()
    try:
        p.wait(90)
    except Exception:
        p.kill()
    time.sleep(10)
