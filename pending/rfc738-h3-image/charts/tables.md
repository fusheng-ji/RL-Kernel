### Parameters by top-level module (analytic table = released headers)

| module | parameters |
|---|---|
| transformer_blocks | 32,278,592,000 |
| token_refiner | 770,730,752 |
| norm_out | 28,917,504 |
| context_embedder | 27,530,496 |
| time_embedder | 15,835,008 |
| pdd_proj_out | 3,097,152 |
| proj_in | 521,472 |

### Probe I8 (Torch linspace) vs the released grid

| canvases | differ in FP64 | differ after FP32 cast | max abs FP64 |
|---|---|---|---|
| 1419 | 1386 | 35 | 5.68e-14 |

### Canvas plans (requested -> canvas)

| requested | canvas | scale | resize back |
|---|---|---|---|
| 256x256 | 2048x2048 | 8.0000 | True |
| 384x384 | 2048x2048 | 5.3333 | True |
| 512x512 | 2048x2048 | 4.0000 | True |
| 768x256 | 3552x1184 | 4.6188 | True |
| 768x384 | 2912x1440 | 3.7712 | True |
| 768x512 | 2496x1664 | 3.2660 | True |
| 768x768 | 2048x2048 | 2.6667 | True |
| 768x1024 | 1760x2368 | 2.3094 | True |
| 768x1280 | 1600x2656 | 2.0656 | True |
| 768x1536 | 1440x2912 | 1.8856 | True |
| 768x2048 | 1248x3360 | 1.6330 | True |
| 768x2560 | 1120x3744 | 1.4606 | True |
| 768x3072 | 1024x4096 | 1.3333 | True |
| 768x4096 | 896x4736 | 1.1547 | True |
| 1024x1024 | 2048x2048 | 2.0000 | True |
| 1280x1280 | 2048x2048 | 1.6000 | True |

### Consistent-permutation drift (row identity preserved; repeat bitwise on both GPUs)

| dtype | B200 max abs | H100 max abs | bitwise permutations (of 20) |
|---|---|---|---|
| fp64 | 3.33e-16 | 4.44e-16 | 0 / 0 |
| fp32 | 2.38e-07 | 2.38e-07 | 0 / 0 |

### Patch pack / unpack (CUDA, FP32, batch 1)

| latent | rows | pack µs | unpack µs | round trip bitwise |
|---|---|---|---|---|
| 64x64 | 1024 | 249.8 | 24.6 | True |
| 128x128 | 4096 | 23.5 | 24.4 | True |
| 222x296 | 16428 | 26.0 | 27.1 | True |
| 256x256 | 16384 | 26.1 | 26.8 | True |

### Reference conditioning probe arms

| arm | max abs | rows changed | elements changed | guard |
|---|---|---|---|---|
| I12 noise before target | 5.72e-03 | 747 / 747 | 71,712 / 71,712 | tripped |
| I13 clean (t = 1) | 6.59e-03 | 747 / 747 | 71,712 / 71,712 | digest differs |
| I13 schedule t | 6.37e+00 | 747 / 747 | 71,712 / 71,712 | digest differs |
| Python-float level | 2.38e-07 | 747 / 747 | 20,289 / 71,712 | digest differs |
| I13 resample at step 3 | — | — | — | check_fixed trips at step 3 |
| I14 write every row | — | — | — | check_fixed trips at step 0 |

### Text presentation: equality and latency (50-layer stack)

| case | rows | B200 ms | H100 ms | equal Diffusers B200 / H100 |
|---|---|---|---|---|
| t2i | 38 | 25.8 | 53.4 | True / True |
| t2i_truncated | 4096 | 316.2 | 560.7 | True / True |
| edit | 3957 | 359.5 | 655.1 | True / True |
| edit_swapped | 3957 | 356.8 | 654.7 | True / True |

### Sigma grids (FP32)

| index | released | H3 shift-12 linspace | shift_provided_sigmas | timestep (released) |
|---|---|---|---|---|
| 0 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| 1 | 0.967524 | 0.983607 | 0.997211 | 0.032476 |
| 2 | 0.887549 | 0.960000 | 0.989552 | 0.112451 |
| 3 | 0.750000 | 0.923077 | 0.972973 | 0.250000 |
| 4 | 0.532774 | 0.857143 | 0.931896 | 0.467226 |
| 5 | 0.232009 | 0.705882 | 0.783792 | 0.767991 |
| 6 | 0.000000 | 0.000000 | 0.000000 | — |

### Velocity step: programs and latency (16,384 x 96 rows, B200)

| step | sigma -> next | Triton != reference | blend != velocity | max abs (blend) | reference µs | Triton µs |
|---|---|---|---|---|---|---|
| 0 | 1.0000 -> 0.9675 | 0 | 398,556 (25%) | 4.8e-07 | 115 | 136 |
| 1 | 0.9675 -> 0.8875 | 0 | 480,708 (31%) | 4.8e-07 | 117 | 131 |
| 2 | 0.8875 -> 0.7500 | 0 | 499,700 (32%) | 4.8e-07 | 113 | 130 |
| 3 | 0.7500 -> 0.5328 | 0 | 578,859 (37%) | 4.8e-07 | 113 | 131 |
| 4 | 0.5328 -> 0.2320 | 0 | 689,026 (44%) | 4.8e-07 | 113 | 131 |
| 5 | 0.2320 -> 0.0000 | 0 | 0 (0%) | 0.0e+00 | 112 | 129 |

### Phase B chain, released weights

| fixture | request | canvas | sequence (ref rows) | gate | output bytes | run to run | RL-Kernel s | Diffusers s | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| edit_example | 4096x4096 | 4096x4096 | 18,867 (1,024) | True | True | True | 13.8 | 18.7 | 67.2 |
| edit_two_refs | 2048x2048 | 2048x2048 | 17,203 (6,528) | True | True | True | 10.3 | 11.5 | 66.9 |
| t2i_example | 4736x3552 | 4736x3552 | 16,741 (0) | True | True | True | 14.7 | 18.2 | 66.8 |
| t2i_upscaled | 1024x768 | 2368x1760 | 4,383 (0) | True | True | True | 3.0 | 4.2 | 64.7 |
