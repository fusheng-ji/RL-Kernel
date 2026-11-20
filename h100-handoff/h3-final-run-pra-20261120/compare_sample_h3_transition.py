"""h3_sample_previous_step (RL-Kernel PR A) against verl-omni's sample_h3_transition, cps.

For two sigma grids, every step, a CUDA and a CPU generator, the sampling path and the scoring
path (given prev_sample): prev_sample, prev_sample_mean, std_dev_t and sqrt_dt are compared
bitwise (dtype, shape, bytes); log_prob is compared bitwise and by max |diff| (ours is the
fixed-order mean, verl-omni's is Tensor.mean). Run with RL-Kernel and verl-omni importable.

    python compare_sample_h3_transition.py --out report.json
"""

import argparse, json, subprocess
import torch
import diffusers, verl_omni
from verl_omni.pipelines.minimax_h3_flow_grpo import common as vo
from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler
from rl_engine.models.minimax_h3 import transition as tr


def same(a, b):
    """Same dtype, shape and bytes."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    dev = torch.device("cuda")
    grids = {"linear_30": torch.linspace(1.0, 0.0, 31)}
    s = torch.linspace(1.0, 0.0, 41)[:-1]; shift = 5.0
    grids["shift5_40"] = torch.cat([shift * s / (1 + (shift - 1) * s), torch.zeros(1)])
    shapes = {"video": (3, 24, 5, 16, 16), "audio": (3, 40, 32)}
    rows, worst_logp = [], 0.0
    for gname, sig in grids.items():
        sched = FlowMatchSDEDiscreteScheduler(); vo.configure_flow_scheduler(sched, sig, dev)
        for step in range(len(sched.timesteps)):
            idx = sched.index_for_timestep(sched.timesteps[step])
            for mod, shp in shapes.items():
                g0 = torch.Generator(device="cpu").manual_seed(1000 + step)
                x = torch.randn(shp, generator=g0).to(dev); v = torch.randn(shp, generator=g0).to(dev)
                for gdev in ("cuda", "cpu"):
                    for path in ("sample", "score"):
                        prev = None if path == "sample" else (x + 0.01 * v).contiguous()
                        ga = torch.Generator(device=gdev).manual_seed(7 + step); gb = torch.Generator(device=gdev).manual_seed(7 + step)
                        sched._step_index = None
                        theirs = vo.sample_h3_transition(sched, x, v, step, noise_level=tr.NOISE_LEVEL, sde_type="cps", generator=ga, prev_sample=prev)
                        ours = tr.h3_sample_previous_step(x, v, sched.sigmas, idx, noise_level=tr.NOISE_LEVEL, sde_type="cps", generator=gb, prev_sample=prev)
                        names = ("prev_sample", "log_prob", "prev_sample_mean", "std_dev_t", "sqrt_dt")
                        r = {"grid": gname, "step": step, "modality": mod, "generator": gdev, "path": path}
                        for n, a, b in zip(names, ours, theirs):
                            r[n] = same(a, b)
                        d = (ours[1].double() - theirs[1].double()).abs().max().item()
                        r["log_prob_max_abs"] = d; worst_logp = max(worst_logp, d)
                        rows.append(r)
    keys = ("prev_sample", "prev_sample_mean", "std_dev_t", "sqrt_dt", "log_prob")
    summary = {k: f"{sum(r[k] for r in rows)}/{len(rows)}" for k in keys}
    rep = {"torch": torch.__version__, "diffusers": diffusers.__version__, "verl_omni_commit": subprocess.run(["git", "-C", verl_omni.__path__[0], "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "rl_kernel_commit": subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "device": torch.cuda.get_device_name(0), "noise_level": tr.NOISE_LEVEL, "cases": len(rows), "bitwise_equal": summary, "log_prob_max_abs": worst_logp, "rows": rows}
    open(args.out, "w").write(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("torch", "diffusers", "verl_omni_commit", "rl_kernel_commit", "cases", "bitwise_equal", "log_prob_max_abs")}))


if __name__ == "__main__":
    main()
