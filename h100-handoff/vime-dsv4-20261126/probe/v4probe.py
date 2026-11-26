"""Validation hooks for the DeepSeek V4 run, loaded via --custom-rm-path / --rollout-data-postprocess-path.
The author's checkout is not modified. postprocess() runs in every trainer rank before each train step."""
import gc, json, os, socket, time
import torch

OUT = os.environ.get("V4PROBE_DIR", "/tmp")
_state = {"step": 0}


async def reward(args, sample, **kwargs):
    # Non-constant inside a prompt group, so the GRPO advantage is non-zero.
    r = sample.response or ""
    if os.environ.get("V4REWARD") == "lower":
        return sum("a" <= c <= "z" for c in r) / max(len(r), 1)
    return sum(c.isalpha() for c in r) / max(len(r), 1)


def _log(rec):
    rec = {"t": time.time(), "host": socket.gethostname(), "rank": int(os.environ.get("RANK", -1)), **rec}
    line = json.dumps(rec)
    print("V4PROBE " + line, flush=True)
    with open(os.path.join(OUT, f"probe-rank{rec['rank']}.jsonl"), "a") as f:
        f.write(line + "\n")


def _actor():
    for o in gc.get_objects():
        try:
            if type(o).__name__ in ("MegatronTrainRayActor", "Class") and hasattr(o, "optimizer") and hasattr(o, "model"):
                return o
        except Exception:  # noqa: BLE001
            continue
    return None


def _named(actor):
    for chunk in actor.model:
        m = chunk
        while hasattr(m, "module"):
            m = m.module
        yield from m.named_parameters()


def _sample(p):
    t = p.detach()
    return t[:256, :1024] if t.dim() == 2 else t.reshape(-1)[:262144]


def _snap(actor):
    return {n: _sample(p).float().cpu().clone() for n, p in _named(actor) if p.is_floating_point()}


QUANT = ("linear_q_down", "linear_q_up", "linear_kv_proj", "linear_o_group", "linear_proj", "experts.linear_fc")


def _fp8(s):
    from vllm.utils.deep_gemm import per_block_cast_to_fp8
    return per_block_cast_to_fp8(s.contiguous().to(torch.bfloat16), block_size=[128, 128], use_ue8m0=True)[0]


def _compare(before, after, tag):
    tot = chg = ftot = fchg = nt = nc = 0
    mx, nonfinite = 0.0, []
    for n, a in after.items():
        b = before[n]
        if not torch.isfinite(a).all():
            nonfinite.append(n)
        d = torch.nan_to_num((a - b).abs())
        c = int((d > 0).sum())
        tot += a.numel(); chg += c; nt += 1; nc += c > 0
        mx = max(mx, float(d.max()) if d.numel() else 0.0)
        if a.dim() == 2 and min(a.shape) >= 128 and any(q in n for q in QUANT) and torch.isfinite(a).all():
            r, k = a.shape[0] // 128 * 128, a.shape[1] // 128 * 128
            wa, wb = _fp8(a[:r, :k].cuda()), _fp8(b[:r, :k].cuda())
            ftot += wa.numel(); fchg += int((wa.view(torch.uint8) != wb.view(torch.uint8)).sum())
    _log({"kind": "weights", "tag": tag, "bf16_changed_frac": chg / max(tot, 1), "tensors": nt, "tensors_changed": nc,
          "max_abs_delta": mx, "fp8_changed_frac": fchg / max(ftot, 1), "fp8_elems": ftot,
          "nonfinite_params": len(nonfinite), "nonfinite_examples": nonfinite[:6]})


def _grad(p):
    for a in ("main_grad",):
        g = getattr(p, a, None)
        if g is not None:
            return g
    return p.grad


def _norm_probe(actor):
    """Look at the exact gradients the optimizer's norm uses, right before the original norm runs."""
    opt = actor.optimizer
    orig = opt.get_grad_norm

    def get_grad_norm(*a, **k):
        recs = []
        for i, o in enumerate(getattr(opt, "chained_optimizers", [opt])):
            try:
                gs = o.get_grads_for_grad_norm()
                from megatron.core.optimizer import clip_grads as cg
                bad = sum(1 for g in gs if not torch.isfinite(g.float()).all())
                ref = sum(float((g.float() ** 2).sum()) for g in gs) ** 0.5
                impl = getattr(cg.l2_norm_impl, "__module__", "") + "." + getattr(cg.l2_norm_impl, "__name__", str(cg.l2_norm_impl))
                buf = torch.zeros(1, dtype=torch.int, device="cuda")
                one = cg.multi_tensor_applier(cg.l2_norm_impl, buf, [gs], False)[0] if gs else None
                recs.append({"opt": i, "cls": type(o).__name__, "n": len(gs), "bad": bad, "dtypes": sorted({str(g.dtype) for g in gs}),
                             "empty": sum(1 for g in gs if g.numel() == 0), "ref_norm": ref, "impl": impl,
                             "impl_norm": None if one is None else float(one)})
            except Exception as e:  # noqa: BLE001
                recs.append({"opt": i, "err": repr(e)[:200]})
        _log({"kind": "norm", "step": _state["step"], "per_optimizer": recs})
        out = orig(*a, **k)
        _log({"kind": "norm_result", "step": _state["step"], "value": str(out)[:80]})
        return out
    opt.get_grad_norm = get_grad_norm


def _wrap(actor):
    opt = actor.optimizer
    _norm_probe(actor)
    orig = opt.step

    def step(*a, **k):
        bad, n_grad, sq, zero = [], 0, 0.0, 0
        for n, p in _named(actor):
            g = _grad(p)
            if g is None:
                continue
            n_grad += 1
            f = g.detach().float()
            if not torch.isfinite(f).all():
                bad.append((n, int((~torch.isfinite(f)).sum()), int(f.numel())))
            else:
                v = float(f.pow(2).sum())
                sq += v
                zero += v == 0.0
        _log({"kind": "grads", "step": _state["step"], "with_grad": n_grad, "nonfinite_tensors": len(bad),
              "zero_grad_tensors": zero, "finite_grad_norm": sq ** 0.5, "bad": bad[:12]})
        out = orig(*a, **k)
        if os.environ.get("V4_QAT") == "1":
            _qat_round(actor)
        _log({"kind": "optimizer", "step": _state["step"], "ret": [str(x)[:80] for x in out] if isinstance(out, tuple) else str(out)[:80]})
        return out
    opt.step = step


def _fq(x):
    """FP8 E4M3 dynamic fake quantization in 128-wide groups along the last dim (straight-through)."""
    if not x.is_floating_point():
        return x
    v = x.detach().float()
    n = v.shape[-1]
    pad = (-n) % 128
    vp = torch.nn.functional.pad(v, (0, pad)) if pad else v
    g = vp.reshape(*vp.shape[:-1], -1, 128)
    scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 448.0
    if os.environ.get("V4_FQ_POW2") == "1":
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    q = ((g / scale).to(torch.float8_e4m3fn).float() * scale).reshape(vp.shape)[..., :n]
    q = q.to(x.dtype)
    return x + (q - x).detach()


def _install_fq(actor):
    mode = os.environ["V4_FQ"]
    exp = ("experts.linear_fc1", "experts.linear_fc2", "experts.local_experts")
    att = ("linear_q_down_proj", "linear_q_up_proj", "linear_kv_proj", "linear_proj")
    n = 0
    for chunk in actor.model:
        for name, mod in chunk.named_modules():
            is_exp = ".mlp." in name and (name.endswith("linear_fc1") or name.endswith("linear_fc2"))
            is_att = ".self_attention." in name and name.rsplit(".", 1)[-1] in att
            if (is_exp and mode in ("experts", "all")) or (is_att and mode in ("attn", "all")):
                def pre(m, a):
                    return (_fq(a[0]),) + tuple(a[1:])
                mod.register_forward_pre_hook(pre); n += 1
    _log({"kind": "fq", "mode": mode, "modules": n})


def postprocess(args, rollout_id, rollout_data, *a, **k):
    actor = _state.get("actor")
    if actor is None:
        actor = _state["actor"] = _actor()
        if actor is None:
            _log({"kind": "error", "msg": "actor not found"}); return rollout_data
        _wrap(actor)
        if os.environ.get("V4_FQ"):
            _install_fq(actor)
        _state["base"] = _state["prev"] = _snap(actor)
        _log({"kind": "start", "params": len(_state["base"])})
    else:
        now = _snap(actor)
        _compare(_state["prev"], now, f"step{_state['step']}")
        _compare(_state["base"], now, f"cum{_state['step']}")
        _state["prev"] = now
        _state["step"] += 1
    _log({"kind": "mem", "step": _state["step"], "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30})
    return rollout_data


# --- test-only: send BF16 weights that were rounded to the FP8 E4M3/UE8M0 grid -----------------
# Separates "FP8 weight rounding at sync" from "FP8 activation quantization in the rollout engine":
# the engine runs a BF16 checkpoint (no activation quantization) but receives FP8-rounded weights.
def _round_to_fp8(t):
    from vllm.utils.deep_gemm import per_block_cast_to_fp8
    m, n = t.shape
    pm, pn = (-m) % 128, (-n) % 128
    x = torch.nn.functional.pad(t.detach().to(torch.bfloat16), (0, pn, 0, pm)).contiguous()
    q, sc = per_block_cast_to_fp8(x, block_size=[128, 128], use_ue8m0=True)
    M, N = x.shape
    d = q.float().view(M // 128, 128, N // 128, 128) * sc.float().view(M // 128, 1, N // 128, 1)
    return d.reshape(M, N)[:m, :n].to(torch.bfloat16).contiguous()


def _install_round_sync():
    import vime.backends.megatron_utils.megatron_to_hf as mh
    if getattr(mh, "_v4_round", False):
        return
    mh._v4_round = True

    def rounded(named_tensors, quantization_config):
        out, n = [], 0
        for name, t in named_tensors:
            q = name.endswith(".weight") and (
                any(f".attn.{p}.weight" in name for p in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "indexer.wq_b"))
                or ".ffn.experts." in name or ".ffn.shared_experts." in name)
            out.append((name, _round_to_fp8(t) if q and t.dim() == 2 else t)); n += bool(q)
        _state["rounded"] = _state.get("rounded", 0) + n
        return out
    mh.quantize_deepseek_v4_weights = rounded
    print("V4PROBE round-sync installed", flush=True)


if os.environ.get("V4_ROUND_SYNC") == "1" and "RANK" in os.environ:
    _install_round_sync()


# --- test-only QAT: FP32 main params keep accumulating, the model params used by the forward (and by
# weight sync) are put back on the FP8 grid after every optimizer step (straight-through estimator).
QAT_NAMES = ("self_attention.linear_q_down_proj.weight", "self_attention.linear_q_up_proj.weight",
             "self_attention.linear_kv_proj.weight", "self_attention.linear_o_group_proj",
             "self_attention.linear_proj.weight", "core_attention.indexer.linear_wq_b.weight",
             "mlp.experts.", "mlp.shared_experts.")


def _qat_round(actor):
    n = changed = total = 0
    with torch.no_grad():
        for name, p in _named(actor):
            if p.dim() != 2 or min(p.shape) < 128 or not any(q in name for q in QAT_NAMES):
                continue
            r = _round_to_fp8(p).to(p.dtype)
            changed += int((r != p).sum()); total += p.numel(); n += 1
            p.data.copy_(r)
    _log({"kind": "qat", "step": _state["step"], "tensors": n, "elements_moved_frac": changed / max(total, 1)})
