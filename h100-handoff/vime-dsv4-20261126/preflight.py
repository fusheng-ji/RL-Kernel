"""V2a preflight on one GPU. Prints one JSON object; each check is independent and records its error.

- capability / device: expected [9, 0] on H100.
- deep_gemm: what the pinned vLLM decides on this GPU (B200: supported, E8M0 scales used).
- cudnn_dsa: whether the cuDNN DSA entry point that --dsa-kernel-backend cudnn uses exists (it is not
  called here; T0-50 in V3 is the first real use).
- fp8_grid: the probe's rounding (V4_ROUND_SYNC / V4_QAT) must land on the same grid as vime's sync
  export (quant_weight_ue8m0), dequantized and compared bitwise.
- ckpt: the checkpoint's quantization_config.
"""
import inspect
import json
import sys
import traceback

import torch

out = {}


def check(name):
    def deco(f):
        try:
            out[name] = f()
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": repr(e)[:300]}
            traceback.print_exc()
        return f
    return deco


@check("device")
def _():
    return {"capability": list(torch.cuda.get_device_capability()), "name": torch.cuda.get_device_name(),
            "torch": torch.__version__, "cuda": torch.version.cuda}


@check("deep_gemm")
def _():
    from vllm.utils import deep_gemm as dg
    return {"supported": bool(dg.is_deep_gemm_supported()), "e8m0_used": bool(dg.is_deep_gemm_e8m0_used())}


@check("cudnn_dsa")
def _():
    from cudnn import DSA
    has = hasattr(DSA, "indexer_forward_top_k_wrapper")
    r = {"wrapper": has, "signature": str(inspect.signature(DSA.indexer_forward_top_k_wrapper)) if has else None}
    try:
        import cudnn
        r["cudnn_backend_version"] = cudnn.backend_version()
    except Exception as e:  # noqa: BLE001
        r["cudnn_backend_version"] = repr(e)[:100]
    return r


@check("fp8_grid")
def _():
    from v4probe import _round_to_fp8
    from vime.backends.megatron_utils.vllm import quant_weight_ue8m0
    g = torch.Generator(device="cuda").manual_seed(0)
    res = {}
    for shape in ((384, 640), (300, 520)):
        x = (torch.randn(*shape, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        pm, pn = (-shape[0]) % 128, (-shape[1]) % 128
        xp = torch.nn.functional.pad(x, (0, pn, 0, pm)).contiguous()
        w, s = quant_weight_ue8m0(xp, [128, 128])
        M, N = xp.shape
        ref = (w.float().view(M // 128, 128, N // 128, 128) * s.float().view(M // 128, 1, N // 128, 1)).reshape(M, N)
        ref = ref[: shape[0], : shape[1]].to(torch.bfloat16)
        got = _round_to_fp8(x)
        scales_pow2 = bool(torch.all(torch.log2(s.float()) == torch.round(torch.log2(s.float()))))
        res["x".join(map(str, shape))] = {"bitwise_equal": bool(torch.equal(ref, got)), "scales_pow2": scales_pow2}
    return res


@check("ckpt")
def _():
    c = json.load(open(sys.argv[1] + "/config.json"))
    return {"quantization_config": c.get("quantization_config"), "num_hidden_layers": c.get("num_hidden_layers")}


print(json.dumps(out, indent=1))
