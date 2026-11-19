| Case | RL-Kernel contract (per row) | Diffusers batched, default flags | Diffusers batched, pinned flags |
|---|---|---|---|
| keyframe encode, batch 2 row 0 | = 0, bitwise | 0.0226 | 0.0226 |
| keyframe encode, batch 2 row 1 | = 0, bitwise | 0.0255 | 0.0255 |
| keyframe encode, batch 3 row 0 | = 0, bitwise | 0.0218 | 0.0218 |
| keyframe encode, batch 3 row 1 | = 0, bitwise | 0.0254 | 0.0254 |
| keyframe encode, batch 3 row 2 | = 0, bitwise | 0.0199 | 0.0199 |
| 22-frame encode, batch 2 row 0 | = 0, bitwise | 0.00805 | 0.00805 |
| 22-frame encode, batch 2 row 1 | = 0, bitwise | 0.00621 | 0.00621 |
| 22-frame decode, batch 2 row 0 | = 0, bitwise | = 0, bitwise | = 0, bitwise |
| 22-frame decode, batch 2 row 1 | = 0, bitwise | = 0, bitwise | = 0, bitwise |

Source: `evidence-h100.json` (`checks.batch_invariance`), commit `bbdddc67`, NVIDIA H100.
