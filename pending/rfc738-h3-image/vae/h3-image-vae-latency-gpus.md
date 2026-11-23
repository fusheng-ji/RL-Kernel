| Step | Path | NVIDIA B200 (median ms) | NVIDIA H100 (median ms) | NVIDIA H100 / NVIDIA B200 |
|---|---|---|---|---|
| reference encode | RL-Kernel contract (tiled) | 264.4 | 344.1 | 1.30x |
| reference encode | Diffusers blocks (tiled, default flags) | 256.9 | 342.4 | 1.33x |
| reference encode | Diffusers blocks (tiled, pinned flags) | 256.9 | 342.5 | 1.33x |
| reference encode | contract, tiling off (not the release) | 103.0 | 141.1 | 1.37x |
| decode | RL-Kernel contract (tiled) | 1485.5 | 2150.0 | 1.45x |
| decode | Diffusers blocks (tiled, default flags) | 1504.2 | 2156.9 | 1.43x |
| decode | Diffusers blocks (tiled, pinned flags) | 1496.8 | 2156.4 | 1.44x |
| decode | contract, tiling off (not the release) | 860.2 | 1168.0 | 1.36x |

Source: `evidence-b200-a8f05c22.json` (`timing`), commit `a8f05c22`, NVIDIA B200; `evidence-h100-a8f05c22.json` (`timing`), commit `a8f05c22`, NVIDIA H100.
