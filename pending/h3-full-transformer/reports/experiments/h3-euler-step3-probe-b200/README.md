# B200 step-3 probe (seed 33) for #571

Measured on NVIDIA B200 (sm100), driver 580.126, CUDA 13.0, torch 2.13.0+cu130, clean tree at `eb7840bb`
(the #571 head), pinned `MiniMax-H3@42ed227` weights. Probe script: `h3_dual_trajectory_step3_probe.py` from the
handoff branch (`ce882862`, blob and sha256 are in `seed33_report.json`). It reuses the saved strict single and batched
tensors of the B200 dual-trajectory run (`h3-full-transformer-b200/dual_trajectory_seed33.json` is the summary of that run)
and runs only the step-3 forward.

| arm | step-3 entry latent (single / batched) | max abs velocity difference, video | audio |
|---|---|---|---|
| A_replay | each layout's own step-2 latent | 0.6215 | 0.3052 |
| B_shared_single | both use the single layout's step-2 latent | 4.29e-06 | 0 |
| C_shared_batched | both use the batched layout's step-2 latent | 5.42e-06 | 0 |
| D | difference of the two step-2 latents | video 1.91e-06, audio 0 | |

A reproduces the 0.6215 of the full run.

## What it shows, and what it does not
- The two step-2 latents differ by 1.9e-06 (video) and are identical (audio). The step 0-2 differences were not amplified by the
  Euler updates: the input of step 3 is still about 2e-06 apart.
- That 2e-06 input difference becomes a 0.62 video and 0.31 audio velocity difference in the step-3 forward (the audio latent is
  identical in both layouts, so the audio difference comes from the video side through the joint attention).
- With the same input in both layouts (B, C) the two layouts agree to 4e-06 to 5e-06. The layouts themselves do not produce the
  large difference.
- `conclusion_b200_style` (`supports_latent_accumulation`) is the label the script's threshold rule produces. It should not be read
  as "the Euler integration amplifies the drift": the latent drift stays at the 1e-06 level. The data are consistent with the step-3
  forward being highly sensitive to a very small input difference on this device.
- One seed, one run, one device. The layer where the amplification happens was not located. H100 was not run with this script here.

## Comparison with the H100 run of the same script (`h3-euler-step3-probe-h100/seed33_report.json`, not rerun here)
- Natural step-2 latent difference: 2.62e-06 (H100) against 1.91e-06 (B200); step-3 velocity difference with each layout's own
  latent: 5.7e-06 (H100) against 0.6215 (B200).
- The H100 file's delta-scale sweep multiplies the batched step-2 latent difference by K. The step-3 video velocity difference there is
  5.7e-06 at K=1, 0.53 at K=1000, 3.8 at K=1e6. So the H100 forward also amplifies an input difference, but a difference about a
  thousand times larger than the natural one is needed to reach the size that the B200 shows at K=1. That sweep was not run on B200.

This file replaces the placeholder `seed33_report.json` that was staged here earlier (`arms_B_C_not_executed_on_this_cluster`, arm A
replayed from the historical JSON); arms B and C and a fresh arm A were executed on a B200 for this version.
