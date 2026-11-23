| Row | Kind | max \|Δ\| |
|---|---|---|
| run-to-run: reference encode | property | = 0, bitwise |
| run-to-run: decode uint8 (I-C10) | property | = 0, bitwise |
| vs Diffusers encode step, CPU generator | property | = 0, bitwise |
| vs Diffusers encode step, CUDA generator | property | = 0, bitwise |
| vs Diffusers decode step, uint8 | property | = 0, bitwise |
| vs Diffusers, default flags (all three) | property | = 0, bitwise |
| batch row vs alone: encode (mode) | property | = 0, bitwise |
| batch row vs alone: encode (shared generator) | property | = 0, bitwise |
| batch row vs alone: decode uint8 | property | = 0, bitwise |
| latent normalize CPU vs CUDA | property | = 0, bitwise |
| latent denormalize CPU vs CUDA | property | = 0, bitwise |
| pixel denormalize CPU vs CUDA | property | = 0, bitwise |
| I11 keep FP32 sample (latent) | ablation probe | 0.00125 |
| I15 scalar latent normalization | ablation probe | 1.91 |
| I15 no latent normalization | ablation probe | 4.06 |
| I16 untiled decode (uint8) | ablation probe | 79 |
| I16 overlap 96 instead of 64 (uint8) | ablation probe | 29 |
| I17 [-1, 1] pixels (latent) | ablation probe | 3.06 |
| I18 round instead of truncate (uint8) | ablation probe | 1 |

Source: `evidence-b200-a8f05c22.json` (`checks`), commit `a8f05c22`, NVIDIA B200.
