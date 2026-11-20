"""Cost of the PR A functions against what they replace, one GPU, CUDA-event timing.

* gradient norm + clip of the LoRA-shaped gradients (700 FP32 tensors, 120.4 M elements):
  torch.nn.utils.clip_grad_norm_ vs clip_grad_norm_exact_; also the norm alone
  (torch.nn.utils.get_total_norm vs exact_norm);
* one H3 transition at the H3 size (video latents (2, 24, 32, 48, 84), audio (2, 207, 32)),
  sampling and scoring: verl-omni sample_h3_transition vs h3_sample_previous_step
  (needs verl-omni importable; skipped otherwise).
Each row: 5 warm-up calls, then --repeats timed calls; median and min in ms.

    python timing_probe.py --out report.json [--repeats 30]
"""
import argparse, json, statistics
import torch
from rl_engine.ops.norm.reproducible import clip_grad_norm_exact_, exact_norm


def timed(fn, repeats):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return {"median_ms": statistics.median(ts), "min_ms": min(ts), "runs": repeats}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--repeats", type=int, default=30); args = ap.parse_args()
    g = torch.Generator().manual_seed(0)
    grads = [(torch.randn(*s, generator=g) * 1e-3).cuda() for s in [(32, 5376), (5376, 32)] * 350]
    ps = [torch.nn.Parameter(torch.zeros_like(t)) for t in grads]
    def reset():
        for p, t in zip(ps, grads): p.grad = t.clone()
    rows = {}
    rows["norm_torch"] = timed(lambda: float(torch.nn.utils.get_total_norm(grads, 2.0)), args.repeats)
    rows["norm_exact"] = timed(lambda: exact_norm(grads), args.repeats)
    reset(); rows["clip_torch"] = timed(lambda: float(torch.nn.utils.clip_grad_norm_(ps, 1e9)), args.repeats)
    reset(); rows["clip_exact"] = timed(lambda: clip_grad_norm_exact_(ps, 1e9), args.repeats)
    try:
        from verl_omni.pipelines.minimax_h3_flow_grpo import common as vo
        from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler
        from rl_engine.models.minimax_h3 import transition as tr
        dev = torch.device("cuda"); sched = FlowMatchSDEDiscreteScheduler(); vo.configure_flow_scheduler(sched, torch.linspace(1.0, 0.0, 31), dev)
        step = 10; idx = sched.index_for_timestep(sched.timesteps[step])
        for mod, shp in {"video": (2, 24, 32, 48, 84), "audio": (2, 207, 32)}.items():
            x = torch.randn(shp, generator=g).to(dev); v = torch.randn(shp, generator=g).to(dev); prev = (x + 0.01 * v).contiguous()
            gen = torch.Generator(device="cuda").manual_seed(1)
            def theirs(p=None):
                sched._step_index = None
                return vo.sample_h3_transition(sched, x, v, step, noise_level=tr.NOISE_LEVEL, sde_type="cps", generator=gen, prev_sample=p)
            def ours(p=None):
                return tr.h3_sample_previous_step(x, v, sched.sigmas, idx, generator=gen, prev_sample=p)
            rows[f"transition_{mod}_sample_verl_omni"] = timed(lambda: theirs(), args.repeats)
            rows[f"transition_{mod}_sample_rl_kernel"] = timed(lambda: ours(), args.repeats)
            rows[f"transition_{mod}_score_verl_omni"] = timed(lambda: theirs(prev), args.repeats)
            rows[f"transition_{mod}_score_rl_kernel"] = timed(lambda: ours(prev), args.repeats)
    except ImportError as e:
        rows["transition_skipped"] = str(e)
    rep = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "rows": rows}
    open(args.out, "w").write(json.dumps(rep, indent=1) + "\n"); print(json.dumps(rep))


if __name__ == "__main__":
    main()
