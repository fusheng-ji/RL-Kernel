| Step (median ms) | RL-Kernel contract (tiled) | Diffusers recipe (tiled, default flags) | Diffusers recipe (tiled, pinned flags) | contract, tiling off (not the release) |
|---|---|---|---|---|
| keyframe encode | 97.5 | 97.5 | 97.5 | 56.9 |
| 22-frame encode | 1852.1 | 1853.7 | 1851.0 | — |
| 22-frame decode | 602.0 | 491.3 | 490.3 | 469.4 |

Source: `evidence-h100.json` (`timing`), commit `bbdddc67`, NVIDIA H100.
