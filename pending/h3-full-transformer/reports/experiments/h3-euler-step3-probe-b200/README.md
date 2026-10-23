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
- That input difference becomes a 0.62 video and 0.31 audio velocity difference in the step-3 forward (the audio latent is
  identical in both layouts, so the audio difference comes from the video side through the joint attention). What the 2e-06
  amounts to after the bfloat16 cast is in the correction below.
- With the same input in both layouts (B, C) the two layouts agree to 4e-06 to 5e-06. The layouts themselves do not produce the
  large difference.
- `conclusion_b200_style` (`supports_latent_accumulation`) is the label the script's threshold rule produces. It should not be read
  as "the Euler integration amplifies the drift": the latent drift stays at the 1e-06 level. See the correction below for what the
  data do show about the size of the input difference.
- One seed, one run, one device. The layer where the amplification happens was not located. H100 was not run with this script here.

## Correction (added after the first version of this file)
The first version said the B200 was about a thousand times more sensitive to the input difference than the H100. That is **withdrawn**.
The model casts the FP32 latent to bfloat16 before `proj_in`, so a 1e-06 FP32 difference becomes either no difference or a full
bfloat16 step. Counting the elements that differ after the cast in the saved B200 tensors (single vs batched latent):

| entry latent | elements differing in FP32 (video) | elements differing after the bf16 cast (video) | largest difference after the cast |
|---|---|---|---|
| seed 33, step 2 (the input of step 3) | 306 of 384 | **1** | 1.95e-03 |
| seed 33, step 0 | 322 of 384 | 1 | 9.5e-07 (an element near zero) |
| seed 33, step 1 | 323 of 384 | 0 | 0 |
| seeds 11 and 22, steps 0 to 2 | 311 to 339 of 384 | 0 | 0 |

The audio latents are identical in every case. So the input of step 3 in seed 33 differs by one bfloat16 step in a single video
element (1.95e-03), and every other entry latent that was checked is bitwise identical after the cast. A 1.95e-03 difference in one
input element giving a 0.62 velocity difference is the same order as the H100 file's sweep (`h3-euler-step3-probe-h100/seed33_report.json`,
not rerun here: about 2.6e-03 on the latents, K=1000, gives 0.53). On this evidence the two devices are not shown to differ in
sensitivity; the difference between the natural runs may be only that the B200 seed-33 latents happen to straddle a bfloat16
rounding boundary. Whether the H100 natural run has such an element can be checked on its saved tensors with the same count (it was
not checked here).

This file replaces the placeholder `seed33_report.json` that was staged here earlier (`arms_B_C_not_executed_on_this_cluster`, arm A
replayed from the historical JSON); arms B and C and a fresh arm A were executed on a B200 for this version.

## Layer by layer (`layerwise_report.json`, script `layerwise_probe.py`)
Same B200 and commit; five forwards of the step-3 request, every block's output (and, for blocks 0 and 1, every graph node) compared:

| arm | forwards compared | step-3 velocity difference, video / audio |
|---|---|---|
| A | (single, own latent) vs (batched, own latent) | 0.6215 / 0.3052 |
| B | same latent in both layouts | 4.3e-06 / 0 |
| E | same layout (single), the two latents | 0.6215 / 0.3052 |
| P | same layout (single), latent identical except the one element that differs after the bf16 cast, set to the other value | 0.6215 / 0.3052 |

- A, E and P agree on every block and node (not only on the final velocity), and B is exactly zero in every block. The single element
  (video latent index `[0,1,8]`: FP32 0.4072261 against 0.4072277, bfloat16 0.40625 against 0.408203) is the whole difference.
- Relative difference (norm of the difference over norm of the reference) of the block outputs in arm P: 7.1e-04 after block 0, 1.2e-03
  after block 1, 5.5e-03 after block 10, 6.8e-03 after block 30, 4.8e-02 after block 39, 5.5e-02 after block 40, 0.22 after block 49: about
  300 times between the first and the last block. Most blocks grow it by 1.1 to 1.3 times; the largest steps are block 39 (1.77 times, where the
  largest activation also rises from about 4e+04 to 2e+05) and block 1 (1.74 times). The fraction of block-output elements that differ is
  20% after block 0 and 99% after block 49.
- In block 0, from the input embedding (2.1e-04) to the block output (7.1e-04) the largest node-level relative difference is at `ffn_down` (2.5e-03).
- The final step-3 velocity differs by at most 0.62 against a reference maximum of 14.6 (video) and 0.31 against 4.8 (audio).

What this does not show: a layer that is especially sensitive (the growth is spread over the depth), or what the H100 forward does with a
perturbation of this exact kind (not run). One seed, one perturbation, one run.
