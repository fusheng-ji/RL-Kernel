#!/usr/bin/env python3
"""Start a TP4 vLLM engine the way qwen3_next_rollout_smoke.py --non-strict-smoke does, on a small
model with dummy weights, and generate 8 tokens. Used to reproduce the H100 abort
(`TypeAttr __ffi_repr__ is already registered for type index 132` right after the FlashInfer
all-reduce norm fusion workspace is created with backend=trtllm) on B200.

Usage: probe.py --model <dir> --out <json> [--no-fuse] [--gdn-triton]
The all-reduce backend comes from VLLM_FLASHINFER_ALLREDUCE_BACKEND (set by the caller).
"""

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-fuse", action="store_true")
    p.add_argument("--gdn-triton", action="store_true")
    a = p.parse_args()
    os.environ["VLLM_BATCH_INVARIANT"] = "0"  # as --non-strict-smoke
    os.environ["VLLM_PLUGINS"] = ""
    from vllm import LLM, SamplingParams

    config = json.loads((a.model / "config.json").read_text())
    kwargs = dict(
        model=str(a.model),
        dtype="bfloat16",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.5,
        max_model_len=512,
        max_num_seqs=2,
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=1234,
        load_format="dummy",
        kernel_config={"moe_backend": "triton", "enable_flashinfer_autotune": False},
    )
    if "ConditionalGeneration" in config["architectures"][0]:
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        kwargs["mamba_cache_dtype"] = "bfloat16"
        kwargs["mamba_ssm_cache_dtype"] = "float32"
    if a.no_fuse:
        kwargs["compilation_config"] = {"pass_config": {"fuse_allreduce_rms": False}}
    if a.gdn_triton:
        kwargs["gdn_prefill_backend"] = "triton"
    llm = LLM(**kwargs)
    pass_config = llm.llm_engine.vllm_config.compilation_config.pass_config
    out = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=8))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "model": a.model.name if a.model.name else str(a.model),
                "architecture": config["architectures"][0],
                "backend_env": os.environ.get("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "auto"),
                "fuse_allreduce_rms": pass_config.fuse_allreduce_rms,
                "gdn_prefill_backend": kwargs.get("gdn_prefill_backend", "default"),
                "tokens": list(out[0].outputs[0].token_ids),
            },
            indent=2,
        )
        + "\n"
    )
    print("PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
