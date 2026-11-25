"""Observe-only capture for the native verl-omni H3 run. Active only with H3CAP_DIR set; changes no numerics.

- ``sample_h3_transition`` is the one transition entry point of rollout and actor. The rollout call (no ``prev_sample``)
  stores velocity / sample / mean under a digest of the next latent it returns; the actor replay (``prev_sample`` given)
  stores its own under the digest of that ``prev_sample``.
- ``dispatch_attention_fn`` (Diffusers) and ``_get_hub_module`` (vLLM-Omni) record which attention backend and which
  kernel file each process used, once per distinct value. A recording failure is printed and never raised.
"""
import os
import sys

_DIR = os.environ.get("H3CAP_DIR")
_SEEN = set()


def _note(kind, value):
    if (kind, value) in _SEEN:
        return
    _SEEN.add((kind, value))
    try:
        import json

        d = os.path.join(_DIR, "backends")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps({"kind": kind, "value": value}) + "\n")
    except Exception as e:
        print("H3CAP note failed:", e, file=sys.stderr, flush=True)


def _verl(mod):
    import hashlib
    import torch

    orig = mod.sample_h3_transition
    calls = {"n": 0}

    def wrapped(scheduler, sample, h3_velocity, step, **kw):
        out = orig(scheduler, sample, h3_velocity, step, **kw)
        if not kw.get("return_log_prob", True) or out[1] is None:
            return out
        prev = kw.get("prev_sample")
        side = "rollout" if prev is None else "actor"
        nxt = out[0] if prev is None else prev
        nxt = nxt.detach().float().contiguous().cpu()
        key = hashlib.sha256(nxt.numpy().tobytes()).hexdigest()[:24]
        calls["n"] += 1
        rec = {"velocity": h3_velocity.detach().float().cpu(), "velocity_dtype": str(h3_velocity.dtype),
               "sample": sample.detach().float().cpu(), "sample_dtype": str(sample.dtype),
               "mean": out[2].detach().float().cpu(), "std": torch.as_tensor(out[3]).detach().float().cpu(),
               "sqrt_dt": torch.as_tensor(out[4]).detach().float().cpu() if len(out) > 4 else None,
               "log_prob": out[1].detach().float().cpu(), "step": int(step), "grad": torch.is_grad_enabled(),
               "call": calls["n"], "pid": os.getpid()}
        d = os.path.join(_DIR, side)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{key}.{os.getpid()}.{calls['n']}.pt")
        torch.save(rec, path + ".tmp")
        os.replace(path + ".tmp", path)
        return out

    mod.sample_h3_transition = wrapped


def _diffusers(mod):
    orig = mod.dispatch_attention_fn

    def wrapped(*a, **kw):
        out = orig(*a, **kw)
        try:
            b = kw.get("backend")
            b = mod._AttentionBackendRegistry.get_active_backend()[0] if b is None else b
            name = str(getattr(b, "value", b))
            if ("actor_attention_backend", name) not in _SEEN:
                _note("actor_attention_backend", name)
                cfg = mod._HUB_KERNELS_REGISTRY.get(mod.AttentionBackendName(name))
                fn = getattr(cfg, "kernel_fn", None)
                if fn is not None:
                    _note("actor_kernel_file", str(fn.__code__.co_filename))
        except Exception as e:
            print("H3CAP actor backend note failed:", e, file=sys.stderr, flush=True)
        return out

    mod.dispatch_attention_fn = wrapped


def _hub(mod):
    orig = mod._get_hub_module

    def wrapped(repo_id):
        m = orig(repo_id)
        _note("rollout_kernel_file", f"{repo_id} {getattr(m, '__file__', '?')}")
        return m

    mod._get_hub_module = wrapped


_HOOKS = {
    "verl_omni.pipelines.minimax_h3_flow_grpo.common": _verl,
    "diffusers.models.attention_dispatch": _diffusers,
    "vllm_omni.diffusion.attention.backends.flash_attn_hub": _hub,
}

if _DIR:
    import importlib.abc
    import importlib.util

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name not in _HOOKS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                _HOOKS[name](module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
