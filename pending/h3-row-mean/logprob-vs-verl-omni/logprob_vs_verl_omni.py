"""The CPS log-prob with the old and the new fixed-order mean, against verl-omni and FP64.

Scoring path (prev_sample given) at the H3 latent sizes, steps 0 / 10 / 29 of a 30-step grid,
a batch of 8: RL-Kernel's cps_transition (this checkout's fixed_order_mean, h3-row-mean-v1),
the same arithmetic with the previous mean (h3-projection-gemm-v1 GEMM [B, K] x [K, 1]),
verl-omni's sample_h3_transition (Tensor.mean) and an FP64 mean of the same squared
differences. Also: is each sample's log-prob the same bytes alone and in batches of 2 / 4 / 8.

    python logprob_vs_verl_omni.py --out report.json
"""
import argparse, json, subprocess
import torch
from verl_omni.pipelines.minimax_h3_flow_grpo import common as vo
from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler
from rl_engine.backends.shared.triton.gemm import h3_projection as g
from rl_engine.models.minimax_h3 import transition as tr

SHAPES = {"video": (24, 32, 48, 84), "audio": (207, 32)}


def gemm_mean(x):
    flat = x.reshape(x.shape[0], -1).contiguous().float()
    return g.gemm(flat, torch.ones(flat.shape[1], 1, device=flat.device))[:, 0] / flat.shape[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    dev = torch.device("cuda"); sched = FlowMatchSDEDiscreteScheduler(); vo.configure_flow_scheduler(sched, torch.linspace(1.0, 0.0, 31), dev)
    rows = []
    for mod, shp in SHAPES.items():
        gen = torch.Generator().manual_seed(11)
        x = torch.randn((8,) + shp, generator=gen).to(dev); v = torch.randn((8,) + shp, generator=gen).to(dev)
        prev = (x + 0.01 * torch.randn((8,) + shp, generator=gen).to(dev)).contiguous()
        for step in (0, 10, 29):
            idx = sched.index_for_timestep(sched.timesteps[step])
            sig, sp = sched.sigmas[idx], sched.sigmas[idx + 1]
            view = (-1,) + (1,) * len(shp)
            s_b, sp_b = sig.expand(8).reshape(view), sp.expand(8).reshape(view)
            ours = tr.cps_transition(x, -v, s_b, sp_b, prev_sample=prev)
            sq = -((prev - ours.prev_sample_mean) ** 2)
            fp64 = sq.double().reshape(8, -1).mean(dim=1)
            old = gemm_mean(sq)
            sched._step_index = None
            theirs = vo.sample_h3_transition(sched, x, v, step, noise_level=tr.NOISE_LEVEL, sde_type="cps", prev_sample=prev)[1]
            rel = lambda a: ((a.double() - fp64).abs() / fp64.abs()).max().item()
            alone = [tr.cps_transition(x[i:i+1], -v[i:i+1], s_b[:1], sp_b[:1], prev_sample=prev[i:i+1]).log_prob for i in range(8)]
            differ = 0
            for b in (2, 4, 8):
                for s in range(0, 8, b):
                    out = tr.cps_transition(x[s:s+b], -v[s:s+b], s_b[:b], sp_b[:b], prev_sample=prev[s:s+b]).log_prob
                    differ += sum(int(not torch.equal(out[j:j+1].view(torch.int32), alone[s + j].view(torch.int32))) for j in range(b))
            rows.append({"modality": mod, "step": step,
                         "new_mean_rel_err_vs_fp64": rel(ours.log_prob), "old_mean_rel_err_vs_fp64": rel(old), "verl_omni_rel_err_vs_fp64": rel(theirs),
                         "new_vs_verl_omni_max_abs": (ours.log_prob.double() - theirs.double()).abs().max().item(),
                         "old_vs_verl_omni_max_abs": (old.double() - theirs.double()).abs().max().item(),
                         "new_bitwise_verl_omni": int(sum(torch.equal(ours.log_prob[i].view(torch.int32), theirs[i].view(torch.int32)) for i in range(8))),
                         "new_differ_from_alone_of_24": differ})
    rep = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
           "rl_kernel_commit": subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip(), "rows": rows}
    open(args.out, "w").write(json.dumps(rep, indent=1) + "\n")
    for r in rows: print(json.dumps({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in r.items()}))


if __name__ == "__main__":
    main()
