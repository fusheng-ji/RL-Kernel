# `feat/h3-pack-noise-rows`: `packed_index_copy` in the forward, `conditioning_noise_mix` in the pipeline

Evidence for the PR on branch `feat/h3-pack-noise-rows` (head `474a71e8`, base `test-h3` `12e81c36`). All files are redacted: home, data and environment paths, user, host, cluster and job identifiers are replaced by placeholders.

| Path | What |
|---|---|
| `b200/pr3-gpu.txt` | `test_h3_packed_forward.py` and `test_h3_conditioning_noise.py` on one B200 (`-v`): the CUDA rows against the provider, 27 passed |
| `b200/suite-pack-noise-rows.txt`, `suite-pack-noise-base.txt` | `tests/models/minimax_h3` at head and at base, 4 GPUs each; `failed-*.txt` list the failing tests (the same one on both sides) |
| `b200/commits.txt`, `env-*.txt`, `gpus.txt`, `summary.txt` | trees, environment (torch, Diffusers, compute capability), GPUs, the job summary |

The timing of both rows against the provider is in `../h3-parallel-v2-gemm-rows/b200/bench/bench.json` (`pack_noise`).
