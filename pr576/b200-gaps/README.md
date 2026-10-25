# PR #576 on B200: tests and timing at the PR head

Head `9f4d0756fdaf1d20e446760948f66bce7ad43be1` (clean tree, extension built `TORCH_CUDA_ARCH_LIST=10.0 RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1`), one NVIDIA B200, driver 580.126.20, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1. Runner: [run.sh](run.sh). Environment headers are in `results/environment.txt`.

| Run | passed | failed | skipped |
|---|---:|---:|---:|
| `test_h3_flash_attention.py` + `test_h3_full_attention.py` (#553) | 47 | 0 | 0 |
| `test_h3_flash_attention.py` alone | 11 | 0 | 0 |
| first attempt without the strict build (`results-nostrict-build/`) | 38 | 0 | 9 (#553 tests: "requires the optional strict NVIDIA CUDA extension"; the bench then has no #553 column) |

The 11 flash tests include FP64 accuracy, bitwise repeat, batch 3 vs 1, head-shard invariance, the FP32-output delta regression and the 2**31-element rejection.

Bench (`tools/validation/models/h3_flash_attention_bench.py`, 3 runs, median ms [min-max]; raw `results/bench-run{1,2,3}.json`):

| tokens | flash fwd / fwd+bwd | #553 fwd / fwd+bwd | SDPA fwd / fwd+bwd | flash vs #553 (fwd+bwd) | flash vs SDPA (fwd+bwd) |
|---|---|---|---|---|---|
| 12 | 0.054 / 0.60 | 0.082 / 0.93 | 0.026 / 0.47 | 1.5x faster | 1.3x slower |
| 925 | 0.168 / 1.05 | 4.32 / 15.46 | 0.054 / 0.45 | 14.7x faster | 2.3x slower |
| 4096 | 1.70 / 9.16 | 86.2 / 282.9 | 0.369 / 1.50 | 30.9x faster | 6.1x slower |
| 8349 | 7.29 / 40.73 | 358.5 / 1173 | 1.41 / 5.68 | 28.8x faster | 7.2x slower |

Relative error vs FP64 at 925 tokens: flash 1.66e-3 (out/dq/dk/dv), #553 1.66e-3, SDPA 2.3e-3 to 2.4e-3 (same as the PR body's H100 numbers).

Draft addition for the PR body Results (B200 section, next to the H100 one):

> **B200** (one GPU, head `9f4d0756`, strict build, torch 2.13): tests 47 passed with the #553 suite (11 flash-only), 0 skipped. Flash is 14.7x / 30.9x / 28.8x faster than #553 in fwd+bwd at 925 / 4096 / 8349 tokens, but 2.3x / 6.1x / 7.2x slower than SDPA, so the "1.6x to 2.3x slower than SDPA" statement holds for the H100 table only. Relative error vs FP64 at 925 tokens is unchanged (flash 1.66e-3, SDPA 2.3e-3). Raw data: `pr_assets` `pr576/b200-gaps/` @ `<EVIDENCE_SHA>`.
