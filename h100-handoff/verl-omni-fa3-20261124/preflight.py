"""Preflight (1 H100): which FA3 kernel each side loads, and whether it runs at H3's attention shapes.

Uses the calls the recipe makes: the actor through Diffusers' ``_flash_3_varlen_hub`` backend, the rollout through
vLLM-Omni's ``FLASH_ATTN_3_HUB``. Prints JSON; nothing is trusted or installed beyond what those two calls load.
``--no-kernel`` checks versions and devices only (non-Hopper GPUs).
"""
import json
import sys

import diffusers
import torch
import transformers
import vllm
import vllm_omni

out = {"torch": torch.__version__, "cuda": torch.version.cuda, "diffusers": diffusers.__version__, "vllm": vllm.__version__,
       "vllm_omni": vllm_omni.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
       "capability": list(torch.cuda.get_device_capability(0)), "memory_MiB": torch.cuda.get_device_properties(0).total_memory >> 20}
from importlib.metadata import PackageNotFoundError, version

for pkg in ("kernels", "peft", "ray", "verl", "imagebind", "pytorchvideo", "ftfy", "iopath", "timm", "soundfile", "TransferQueue"):
    try:
        out[pkg] = version(pkg)
    except PackageNotFoundError:
        out[pkg] = None
import imagebind.models.imagebind_model  # noqa: F401  (the reward's import)
from transformers import ClapModel  # noqa: F401

if "--no-kernel" not in sys.argv:
    from diffusers.models import attention_dispatch as ad
    from vllm_omni.diffusion.attention.backends.flash_attn_hub import _get_hub_module

    b = ad.AttentionBackendName._FLASH_3_VARLEN_HUB
    ad._check_attention_backend_requirements(b)
    ad._maybe_download_kernel_for_backend(b)
    cfg = ad._HUB_KERNELS_REGISTRY[b]
    hub = _get_hub_module(cfg.repo_id)
    out["actor_kernel_file"] = cfg.kernel_fn.__code__.co_filename
    out["rollout_kernel_file"] = hub.flash_attn_varlen_func.__code__.co_filename
    out["same_kernel_file"] = out["actor_kernel_file"] == out["rollout_kernel_file"]
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 3957, 56, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    def actor():
        return ad.dispatch_attention_fn(q, k, v, backend=b)

    def rollout():
        o = hub.flash_attn_func(q, k, v)
        return o[0] if isinstance(o, tuple) else o

    with torch.no_grad():
        a1, a2, r1, r2 = actor(), actor(), rollout(), rollout()
        ref = torch.nn.functional.scaled_dot_product_attention(*(x.transpose(1, 2) for x in (q, k, v))).transpose(1, 2)
    rel = lambda x, y: float((x.double() - y.double()).norm() / y.double().norm())  # noqa: E731
    out["fa3"] = {"actor_repeat_bitwise": torch.equal(a1, a2), "rollout_repeat_bitwise": torch.equal(r1, r2),
                  "actor_equals_rollout_bitwise": torch.equal(a1, r1), "actor_vs_rollout_rel_l2": rel(a1, r1),
                  "actor_vs_torch_sdpa_rel_l2": rel(a1, ref)}
print(json.dumps(out, indent=1))
