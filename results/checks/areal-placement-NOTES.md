# Native fused-RoPE and placement runs

Qwen3-8B / GSM8K, 2 x 4 H100 per job, Megatron TP4 actor and vLLM TP4 rollout
(CUDA graphs). "Driver node" = the node running the verify driver, the AReaL
controller and the Ray head. Node names are redacted to `node-NNN`.

| directory | arm / mode | layout | result |
| --- | --- | --- | --- |
| `r3-ropefusion/S3-sync-ropefusion` | native, fused RoPE, sync, observe | actor on driver node | train 15.20, recompute 7.58, step 38.62 s |
| `r3-ropefusion/S3-profile-ropefusion` | native, fused RoPE, profile | actor on driver node | profiler trace |
| `r3-ropefusion/S4-async-ropefusion` | native, fused RoPE, async | actor alone, vLLM on driver node | train 5.37, recompute 2.36, step 16.38 s |
| `r3-nodecheck/S3-sync-ropefusion-B` | native, fused RoPE, sync | actor alone, vLLM on driver node | train 5.34, recompute 2.26, step 22.33 s |
| `r4-pinned/S3-sync-native-pinned` | native, unfused, sync, observe | actor pinned off driver node | train 8.16, recompute 4.12, step 28.53 s |
| `r4-pinned/S4-async-native-pinned` | native, unfused, async | actor pinned off driver node | train 8.16, recompute 4.33, step 17.75 s |
| `r4-pinned/S3-sync-rlk-pinned` | RL-Kernel round 1, sync | actor pinned off, vLLM on driver node | FAILED: graph capture 117 s, server not healthy in 300 s |
| `r4-pinned/S3-sync-clean-failed-gcs*` | - | - | aborted (Ray head killed by a script bug), ignore |
| `r5-driveroff/S3-sync-rlk-driveroff` | RL-Kernel round 1, sync, 5 steps | driver + GPU-less Ray head on a CPU node | capture 19 s; train 7.01, recompute 2.88, step 26.35 s |

Published (earlier) native unfused sync with the actor on the driver node: train
38.47, recompute 18.08, step 72.67 s.

## Open question

Why is a GPU role slower when it shares a node with the driver / controller /
Ray head? Native actor train 38.47 vs 8.16 s/step; RL-Kernel vLLM graph capture
117 vs 19 s; native vLLM rollout ~1.18x slower.

Ruled out: node hardware. `diagnostics/node_bench.py` and `node_allreduce.py`
give the same numbers on all four nodes: bf16 GEMM ~760-790 TFLOP/s, kernel
launch 7.5 us, H2D ~42 GB/s, CPU loop 1.82 s, TP4 NCCL all-reduce ~640 us at
134 MB.

Observed on a driver node during a run: load average ~1.7, `gpfsSwapd` (GPFS
daemon) at ~37% CPU, `rpc_server` at ~119% CPU, ~4.6k user threads; no obvious
CPU saturation. vLLM warns about an open-file ulimit of 51200.

`diagnostics/restart_ray.sh`, `start_ray.sh`, `run_verify.sh` are the launch
scripts; role pinning uses `RLK_AREAL_RAY_ROLE_RESOURCES=actor=rlk_actor_node`
with the worker raylet started with `--resources='{"rlk_actor_node": 1}'`.
