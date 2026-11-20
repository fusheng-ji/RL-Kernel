"""Does a sample's log_prob depend on the batch it is scored in? verl-omni vs RL-Kernel.

Scoring path (prev_sample given), cps, verl-omni's sample_h3_transition and RL-Kernel's
h3_sample_previous_step on the same inputs. A batch of 8 is scored whole, then every sample is
scored alone and inside batches of 2 and 4 at each position; each sample's log_prob is compared
bitwise with its batch-of-1 value. Video and audio latent shapes at the H3 size.

    python logprob_batch_probe.py --out report.json
"""
import argparse, json
import torch
from verl_omni.pipelines.minimax_h3_flow_grpo import common as vo
from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler
from rl_engine.models.minimax_h3 import transition as tr

SHAPES = {"video": (24, 32, 48, 84), "audio": (207, 32)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    dev = torch.device("cuda")
    sched = FlowMatchSDEDiscreteScheduler(); vo.configure_flow_scheduler(sched, torch.linspace(1.0, 0.0, 31), dev)
    rows = []
    for mod, shp in SHAPES.items():
        g = torch.Generator().manual_seed(11)
        x = torch.randn((8,) + shp, generator=g).to(dev); v = torch.randn((8,) + shp, generator=g).to(dev)
        prev = (x + 0.01 * torch.randn((8,) + shp, generator=g).to(dev)).contiguous()
        for step in (0, 10, 29):
            idx = sched.index_for_timestep(sched.timesteps[step])
            def theirs(sl):
                sched._step_index = None
                return vo.sample_h3_transition(sched, x[sl], v[sl], step, noise_level=tr.NOISE_LEVEL, sde_type="cps", prev_sample=prev[sl])[1]
            def ours(sl):
                return tr.h3_sample_previous_step(x[sl], v[sl], sched.sigmas, idx, prev_sample=prev[sl])[1]
            for name, fn in (("verl_omni", theirs), ("rl_kernel", ours)):
                alone = torch.stack([fn(slice(i, i + 1))[0] for i in range(8)])
                diff, total, worst = 0, 0, 0.0
                for b in (2, 4, 8):
                    for start in range(0, 8, b):
                        out = fn(slice(start, start + b))
                        for j in range(b):
                            total += 1
                            a, c = out[j], alone[start + j]
                            if not torch.equal(a.view(-1).view(torch.int32), c.view(-1).view(torch.int32)):
                                diff += 1; worst = max(worst, (a.double() - c.double()).abs().item())
                rows.append({"implementation": name, "modality": mod, "step": step, "comparisons": total, "differ_from_alone": diff, "max_abs": worst})
    rep = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "shapes": SHAPES, "rows": rows,
           "verl_omni_differ": sum(r["differ_from_alone"] for r in rows if r["implementation"] == "verl_omni"),
           "rl_kernel_differ": sum(r["differ_from_alone"] for r in rows if r["implementation"] == "rl_kernel"),
           "comparisons_each": sum(r["comparisons"] for r in rows if r["implementation"] == "rl_kernel")}
    open(args.out, "w").write(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("torch", "verl_omni_differ", "rl_kernel_differ", "comparisons_each")}))


if __name__ == "__main__":
    main()
