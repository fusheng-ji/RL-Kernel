# B200 pre-check of 1bba6d19 (2026-11-20)

The B200 run named in "What was checked before" has finished. It is a pre-check only; the
H100 run is the result for item 4.

- Hardware: 4 x NVIDIA B200 (183,359 MiB each), driver 580.126.20; torch 2.9.1+cu130.
- Source: `1bba6d19`; `rl_engine._C` loaded from that source tree (`c-file.txt`).
- Model: `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`.
- Rollout manifest `9128f117...`, targets manifest `ccf1c88a...`.

## Host-resident optimizer on the real model

The reference with optimizer state on the CPU equals the one with state on the GPU, bit for bit:

| Pair | Tensors | Differing | Result |
|---|---:|---:|---|
| actor FP32: CPU state vs GPU state | 1604 | 0 | `BITWISE_EQUAL` |
| actor BF16: CPU state vs GPU state | 1604 | 0 | `BITWISE_EQUAL` |

## Peak allocated memory per run (one GPU per HF run)

| Run | Optimizer state | Status | Peak | Time |
|---|---|---|---:|---:|
| frozen targets | - | `frozen_targets_complete` | 30.6 GiB | 38 s |
| actor FP32 | GPU | `fixed_reference_complete` | 131.3 GiB | 724 s |
| actor FP32 | CPU | `fixed_reference_complete` | 70.3 GiB | 815 s |
| actor BF16 | GPU | `fixed_reference_complete` | 157.2 GiB | 696 s |
| actor BF16 | CPU | `fixed_reference_complete` | 65.7 GiB | 848 s |
| actor BF16 implementation control | CPU | `fixed_reference_complete` | 65.7 GiB | 839 s |
| critic FP32 | CPU | `fixed_reference_complete` | 65.7 GiB | 735 s |
| critic BF16 implementation control | CPU | `fixed_reference_complete` | 58.7 GiB | 780 s |

With the state on the CPU, every HF run peaks at 70.3 GiB or less.

## Trainer probes (`--probe`)

| Probe | Status | Peak per rank | Time |
|---|---|---:|---:|
| Megatron, BF16 actor, TP2 (2 ranks) | `probe_complete` | 92.9 GiB | 329 s |
| DeepSpeed, FP32 critic, TP4 (4 ranks) | `probe_complete` | 33.0 GiB | 379 s |

**The Megatron BF16 actor TP2 probe peaks at 92.9 GiB per rank, above the 88 GiB limit of
step 5.** If the H100 run shows the same peak, step 5 stops as written and step 6 is not run.
The other TP2 rows were not probed on B200.

`b9e5690d` follows from this: the Megatron BF16 actor runs at TP4 only, and step 5 probes it at
TP4. Round 2 continues at `19b62661`, which also covers the DeepSpeed BF16 rows; see
`ROUND2.md` for what changed and its B200 probe of all 8 rows.
