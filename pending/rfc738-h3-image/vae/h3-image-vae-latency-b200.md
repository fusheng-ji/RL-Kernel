| Step (median ms) | RL-Kernel contract (tiled) | Diffusers blocks (tiled, default flags) | Diffusers blocks (tiled, pinned flags) | contract, tiling off (not the release) |
|---|---|---|---|---|
| reference encode | 264.4 | 256.9 | 256.9 | 103.0 |
| decode | 1485.5 | 1504.2 | 1496.8 | 860.2 |

Source: `evidence-b200-a8f05c22.json` (`timing`), commit `a8f05c22`, NVIDIA B200.
