# Native diffusers backward: 30-repeat repeatability (H100)

Each cell: distinct bitwise results among 30 identical backward replays; (spread) = max over runs of max|g_i - g_0| / max|g_0|. Largest size per op shown; all smaller sizes in the JSON.

| op | gradient | size | native diffusers | deterministic diffusers | rl-kernel | op's native backward launches atomic `indexFuncLargeIndex` (index_add) |
|---|---|---|---|---|---|---|
| timestep_mlp | `x` | 2048 | 1 (0.0e+00) | 1 | 1 | no |
| timestep_mlp | `w1` | 2048 | 1 (0.0e+00) | 1 | 1 | no |
| timestep_mlp | `b1` | 2048 | 1 (0.0e+00) | 1 | 1 | no |
| timestep_mlp | `w2` | 2048 | 1 (0.0e+00) | 1 | 1 | no |
| timestep_mlp | `b2` | 2048 | 1 (0.0e+00) | 1 | 1 | no |
| adaln_projection | `temb` | 2048 | 1 (0.0e+00) | 1 | 1 | no |
| adaln_projection | `w` | 2048 | 1 (0.0e+00) | 1 | 1 | no |
| adaln_projection | `b` | 2048 | 1 (0.0e+00) | 1 | 1 | no |
| adaln_row_gather | `table` | 32768 | 30 (1.5e-01) | 1 | 1 | yes |
| norm_modulate | `x` | 32768 | 1 (0.0e+00) | 1 | 1 | yes |
| norm_modulate | `w` | 32768 | 1 (0.0e+00) | 1 | 1 | yes |
| norm_modulate | `sh` | 32768 | 30 (1.5e-01) | 1 | 1 | yes |
| norm_modulate | `sc` | 32768 | 30 (1.1e-01) | 1 | 1 | yes |
| gate_residual | `res` | 32768 | 1 (0.0e+00) | 1 | 1 | yes |
| gate_residual | `y` | 32768 | 1 (0.0e+00) | 1 | 1 | yes |
| gate_residual | `gate` | 32768 | 30 (1.5e-01) | 1 | 1 | yes |
| final_adaln_out | `x` | 32768 | 1 (0.0e+00) | 1 | 1 | yes |
| final_adaln_out | `nw` | 32768 | 1 (0.0e+00) | 1 | 1 | yes |
| final_adaln_out | `temb` | 32768 | 30 (7.4e-02) | 1 | 1 | yes |
| final_adaln_out | `w` | 32768 | 30 (2.1e-01) | 1 | 1 | yes |
| final_adaln_out | `b` | 32768 | 30 (1.4e-01) | 1 | 1 | yes |

Environment: torch 2.13.0+cu130, NVIDIA H100. Native runs without `CUBLAS_WORKSPACE_CONFIG`; deterministic runs use `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`, in separate processes. rl-kernel: 1 distinct result for every gradient and size.
