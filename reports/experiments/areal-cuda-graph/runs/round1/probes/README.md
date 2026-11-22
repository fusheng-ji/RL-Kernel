# Single-GPU vLLM probes (Qwen3-8B, TP1, batch 128, 128 prompt + 128 generated tokens)

| file | arm | mode | init s | prefill s | decode ms/step | graph capture |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| native-graph.json | native | graph | 32.1 | 0.402 | 13.11 | 6 s |
| rlkernel-before-graph.json | RL-Kernel before round 1 | graph | 54.5 | 0.544 | 15.66 | 22 s |
| rlkernel-round1-graph.json | RL-Kernel round 1 | graph | 44.3 | 0.538 | 14.58 | 17 s |
| native-eager.json | native | eager | 25.6 | 0.401 | 23.79 | - |
| rlkernel-before-eager.json | RL-Kernel before round 1 | eager | 25.8 | 0.545 | 87.39 | - |
| rlkernel-round1-eager.json | RL-Kernel round 1 | eager | 25.9 | 0.538 | 69.53 | - |

- Capture times come from vLLM's "Graph capturing finished in N secs" log line.
- The rlkernel-round1-graph init includes a hit on the persistent FA4 CuTe JIT
  cache, which saves about 5 s of compilation.
- `rlkernel-init-cprofile-*.txt` holds a cProfile of the RL-Kernel arm's vLLM
  init (profile run and capture), sorted by total time and filtered to
  `rl_engine`.
  - It was taken midway through round 1: after the execute memo and the cos/sin
    reuse, before the token-major RoPE and no-grad GEMM fast paths.
  - cProfile itself stretches the init to about 78 s.
- `fa4_varlen_vs_grouped.json` compares FA4 varlen (packed THD with `cu_seqlens`)
  with the per-length batched launches. Forward output, LSE and dq/dk/dv are
  bitwise equal for 4 random length sets. `bench_ms` is forward plus backward
  of one layer with 25 packed sequences.
