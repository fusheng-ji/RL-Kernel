# H100 results: final heads of #699 / #700 / #701, fast math, extra WS2 layouts (2026-11-14)

Hardware: 2 nodes x 4 H100 (sm90), torch 2.13.0+cu130, CUDA 13.0, triton 3.7.1. Hosts are `node-a` / `node-b`. Paths are redacted. Nothing in the repository trees, test bounds or runners was changed.

## Deviations from the handoff

- **Fast-math runs used a copy of `run_h100.sh` with one changed line.** The original exports `RL_KERNEL_H3_STRICT=1` unconditionally, and the build refuses that together with `KERNEL_ALIGN_USE_FAST_MATH=1` (`ValueError: RL_KERNEL_H3_STRICT is incompatible with KERNEL_ALIGN_USE_FAST_MATH` from `build_tools/envs.py`). The copy reads `RL_KERNEL_H3_STRICT=${STRICT:-1}` and the fast-math runs set `STRICT=0`. This is a finding about the handoff script, not about the PRs.
- **Disk quota.** The project container was full during the first pass. Stages that failed with `Disk quota exceeded` (#700 fulltrans/dual, the fast-math builds, #700 mypy) were deleted and rerun in full after space was freed; no result in this folder is from a failed run.
- **Checkpoint.** `<CKPT>` is the local copy of the 14-shard transformer; its revision (`42ed227`) could not be verified from the files on disk.
- **Peak memory** is not recorded by the WS2 report, so the grid tables have no peak-memory column.

## Item 01: the three heads (default build)

### Tests (passed / skipped / failed)

| PR | Commit | Result | Expected | mypy |
|---|---|---|---|---|
| #699 | `0566cdbb` | 1115 / 3 / 0 | 1115 / 3 / 0 | Found 36 errors in 12 files (36 expected) |
| #700 | `3eda7d4f` | 1122 / 3 / 0 | 1122 / 3 / 0 | Found 36 errors in 12 files (36 expected) |
| #701 | `e42c763a` | 1187 / 18 / 0 | 1187 / 18 / 0 | Found 36 errors in 12 files (36 expected) |

### Full transformer (strict) and provider boundaries

| PR | Run | video | audio | input grads | param grad median / max | candidate bitwise repeatable | provider bitwise repeatable |
|---|---|---|---|---|---|---|---|
| #699 | strict | 2.151e-03 | 2.104e-02 | 0.226 / 0.230 / 0.345 | 0.266 / 0.696 | True | True |
| #699 | provider-boundaries | 2.211e-03 | 2.053e-02 | 0.142 / 0.146 / 0.247 | 0.282 / 0.924 | True | True |
| #700 | strict | 2.511e-03 | 1.543e-02 | 0.118 / 0.156 / 0.248 | 0.176 / 0.434 | True | True |
| #700 | provider-boundaries | 2.211e-03 | 2.053e-02 | 0.142 / 0.146 / 0.247 | 0.282 / 0.924 | True | True |

Expected (strict): #699 video 2.151e-03, audio 2.104e-02, input grads 0.226 / 0.230 / 0.345, param grads 0.266 / 0.696; #700 video 2.511e-03, audio 1.543e-02, input grads 0.118 / 0.156 / 0.248, param grads 0.176 / 0.434. Provider-boundaries of #699 and #700 are expected identical to each other.

`boundary_bindings` (strict run): 
- #699: {'row': 6, 'provider': 2}; non-row entries: ['time_embedder', 'norm_out']
- #700: {'row': 8}
(Expected: #699 six `row` and two `provider` (`time_embedder`, `norm_out`); #700 all eight `row`.)

### Dual trajectory

| PR | seed | strict (differing / compared) | provider (differing / compared) |
|---|---|---|---|
| #699 | 11 | 0 / 16 | 16 / 16 |
| #699 | 22 | 0 / 16 | 16 / 16 |
| #699 | 33 | 0 / 16 | 16 / 16 |
| #700 | 11 | 0 / 16 | 16 / 16 |
| #700 | 22 | 0 / 16 | 16 / 16 |
| #700 | 33 | 0 / 16 | 16 / 16 |

Expected: strict 0/16 and provider 16/16 on every seed.

### #701 grids

| Grid | Size | Distinct per-rank row spans (sizes) | all_bitwise | First drift (forward) | weight_shards_compared | load / canonical / parallel s |
|---|---|---|---|---|---|---|
| tp4-cp2-ring | 40 | 20 / 20 | True | null | [350] | 72.7 / 10.3 / 1.9 |
| tp8-928 | 928 | 1008 | True | null | [350] | 74.2 / 10.5 / 3.0 |
| tp4-cp2-ring-sp-inject | 40 | 5 / 5 / 5 / 5 / 5 / 5 / 5 / 5 | False | {"node": "block.17", "first_row": 16, "owner_ranks": [3]} | [350] | 75.9 / 10.2 / 1.8 |

(The span column lists the distinct per-rank row spans; with SP each TP rank holds its own span.)

Expected: the first two `all_bitwise = true`, 350 shards, `first_drift` null; the SP inject grid `all_bitwise = false` with `first_drift_forward = {node: block.17, first_row: 16, owner_ranks: [3]}`.

## Item 02: fast-math build (`KERNEL_ALIGN_USE_FAST_MATH=1`)

| Tree | Commit | passed / skipped / failed | Failures not in base-fm |
|---|---|---|---|
| base-fm | `07533aee` | 511 / 594 / 0 | none |
| pr699-fm | `0566cdbb` | 524 / 594 / 0 | none |
| pr700-fm | `3eda7d4f` | 531 / 594 / 0 | none |
| pr701-fm | `e42c763a` | 564 / 641 / 0 | none |

Failures in `base-fm` itself: none. The known #569/#571 failure `test_h3_final_adaln_out.py::TestCuda::test_unsupported_device_is_refused_at_construction` did not occur in this build.

## Item 03: #701 extra full-model layouts

| Grid | Size | Distinct per-rank row spans (sizes) | all_bitwise | First drift (forward) | weight_shards_compared | load / canonical / parallel s |
|---|---|---|---|---|---|---|
| tp4-cp2-ulysses | 40 | 20 / 20 | True | null | [350] | 74.0 / 10.4 / 1.8 |
| tp4-cp2-ulysses-925 | 925 | 463 / 462 | True | null | [350] | 79.3 / 12.5 / 2.9 |
| tp4-cp2-ulysses-sp-928 | 928 | 126 / 126 / 126 / 126 / 126 / 126 / 126 / 126 | True | null | [350] | 77.9 / 10.6 / 4.4 |
| tp1-cp3-ring | 40 | 14 / 13 / 13 | True | null | [350] | 72.4 / 10.9 / 3.1 |
| tp1-cp3-ring-925 | 925 | 309 / 308 / 308 | True | null | [350] | 71.9 / 13.1 / 4.8 |
| tp1-cp3-ulysses | | | refused / no report | | | `ValueError: 56 heads do not divide over cp=3` |

