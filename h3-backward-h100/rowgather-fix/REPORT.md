# RowGather: deterministic FP32 backward for diffusers `index_select` gathers (H100)

`RowGather` = `index_select(0, idx)` forward (bitwise unchanged) + backward: stable sort of `idx`, FP32 `segment_reduce` (fixed order, no atomics), cast to BF16. Applied to the unmodified diffusers replay by patching `Tensor.index_select` inside a `with` block (`rowgather_fix.py`). 30 replays per size, same inputs and upstream gradients as `native-repeat30`.

Error = max|err| / max|ref| against FP64 autograd (sizes <= 4097); native error is one run. Size 4097 shown.

| op | gradient | distinct results (30 runs, all sizes) | error fix | error kernel | error native (1 run) | error det diffusers | fix == kernel bitwise |
|---|---|---|---|---|---|---|---|
| adaln_row_gather | `table` | 1 | 2.4e-03 | 2.4e-03 | 5.4e-02 | 5.5e-02 | some sizes |
| norm_modulate | `sh` | 1 | 2.9e-03 | 2.9e-03 | 7.0e-02 | 5.9e-02 | some sizes |
| norm_modulate | `sc` | 1 | 3.5e-03 | 3.5e-03 | 4.8e-02 | 5.7e-02 | some sizes |
| gate_residual | `gate` | 1 | 3.0e-03 | 2.7e-03 | 4.8e-02 | 5.7e-02 | some sizes |
| final_adaln_out | `temb` | 1 | 3.9e-03 | 1.5e-03 | 4.7e-02 | 5.0e-02 | no (all sizes) |
| final_adaln_out | `w` | 1 | 7.2e-03 | 4.3e-03 | 7.6e-02 | 8.0e-02 | no (all sizes) |
| final_adaln_out | `b` | 1 | 4.4e-03 | 3.5e-03 | 5.9e-02 | 6.4e-02 | some sizes |

Forward output bitwise unchanged by the patch: True.

Full-replay backward wall time (ms, includes the rest of the replay, mean of 10):

| op | size | native | fix | kernel |
|---|---|---|---|---|
| adaln_row_gather | 257 | 0.42 | 0.98 | 1.47 |
| adaln_row_gather | 32768 | 11.22 | 11.11 | 4.92 |
| norm_modulate | 257 | 0.53 | 1.59 | 1.51 |
| norm_modulate | 32768 | 7.28 | 8.04 | 3.61 |
| gate_residual | 257 | 0.34 | 0.89 | 1.35 |
| gate_residual | 32768 | 4.42 | 4.87 | 3.07 |
| final_adaln_out | 257 | 0.79 | 1.86 | 1.61 |
| final_adaln_out | 32768 | 7.46 | 9.79 | 3.98 |

## Multi-GPU (4 x H100, one node, NCCL)

Table 42 x 2048 BF16, 32768 tokens. `cp`: tokens sharded over ranks, local table gradient then all_reduce(sum, BF16). `tp`: every rank holds all tokens (replicas). Error vs FP64 sum of the full gather.

| layout | impl | distinct results over 30 repeats (per rank) | ranks identical in every repeat | error range |
|---|---|---|---|---|
| cp | native | [30, 30, 30, 30] | yes | 2.0e-02 to 2.7e-02 |
| cp | fix | [1, 1, 1, 1] | yes | 5.2e-03 to 5.2e-03 |
| tp | native | [30, 30, 30, 30] | NO | 6.0e-02 to 7.4e-02 |
| tp | fix | [1, 1, 1, 1] | yes | 3.3e-03 to 3.3e-03 |

Environment: torch 2.13.0+cu130, NVIDIA H100.
