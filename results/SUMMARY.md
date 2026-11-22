# H100 results: H3 GEMM rows (#535, #554), handoff `c4e3707b` (2026-11-22)

Hardware: 2 nodes (`node-a`, `node-b`), 4 x H100 each, only our processes on the GPUs at the start of every step (`nvidia-smi*.txt`, `other-processes-*.txt`).
Environments: `<E1>` torch 2.13.0+cu130 and `<E2>` torch 2.14.0+cu130 + vLLM 0.33.0 + vLLM-Omni 0.33.0rc2.dev206+g6039984e3; both filtered lists are identical to the last round's (`pip-list-filtered-E1.txt`, `-E2.txt`).

Code: the bundle verifies, and the five `refs/h100/*` commits match `refs.txt` (`code-check/`). Every checkout was a detached worktree, and `git status --porcelain --untracked-files=no` was empty before and after every step (`status-*.txt`). `rl_engine._C.__file__` was inside the checkout before every GPU step (`c-file-*.txt`), except at #731 `f67d96d1` under `<E2>` (deviation 4).

## Verdicts

| Task | Code | Verdict |
|---|---|---|
| H1 | `775e98e7`, base `9388feed` | **FAIL, one finding.** The rows bind and replay as expected, but `test_h3_token_refiner.py::test_backward_against_fp32_golden` fails, only at head. |
| H2.a | `093b0ff2` | PASS |
| H2.b | `093b0ff2`, base `81a87617` | PASS (after a harness relaunch, deviation 3) |
| H3 | `b6e798c8` | PASS |
| H4 | #731 `f67d96d1`; `8b391162` | PASS (#731 ran without `_C`, deviation 4) |
| H5 | #733 `0b5b87c3` | PASS |

### H1. WS1 GEMM rows (`<E1>`, node-a)

**The finding.** `tests/models/minimax_h3/test_h3_token_refiner.py::test_backward_against_fp32_golden` fails at head, both in `binding-tests.txt` (1 failed, 95 passed) and in `suite-head.txt`. It passes at base. The error comes from the O row's operator while the refiner's candidate graph runs backward:

```
rl_engine/validation/models/h3_block.py:233: in cand_gemm
    out = op(flat, weight, logical_keys=_row_keys(x), parameter_id=parameter_id)
self = <rl_engine.ops.gemm.h3_projections.H3AttentionOGemmOp object ...>
parameter_id = 'token_refiner.refiner_blocks.0.attn.to_out.0.weight'
E   RuntimeError: weight gradients require an active canonical backward session
rl_engine/ops/gemm/h3_projections.py:232: RuntimeError
```

The test builds the weights with `requires_grad_()` and calls `hr.run_graph("candidate", ...)` directly with `hb.CandidateOps()`. Its default, `gemm_rows="auto"`, binds the rows on SM90. The test does not open `hb.gemm_session(ops)`, the canonical backward session the rows' weight gradients need. The block's runner opens that session (`rl_engine/validation/models/h3_block.py:859`), and the block tests pass.

Everything else matches the README:
- **Binding tests.** `test_gemm_rows_run_their_operators_on_sm90` ran and passed: nothing was skipped (`-rfs` lists no skips).
- **`block-replay.json`** (tiny 264, small 925, medium 3160), in every layout's `nodes`:
  - `o_proj` is `row` with `H3AttentionOGemmOp`, `ffn_gate_up` is `row` with `H3FFNGateUpGemmOp`, and `ffn_down` is `row` with `H3FFNDownGemmOp`;
  - `attention` is `row` with `H3FullAttentionOp`;
  - `repeat_bitwise_equal` and `batch_row_bitwise_equal` hold for every node, forward and backward. `adaln_projection`'s backward `grad_batch_row_bitwise_equal` is `null`: its input is `temb`, not rows;
  - no `candidate_rel_err_vs_golden` is over the tests' bounds: forward 1.25 x provider + 1e-3; backward leaves 1.5 x provider + 5e-3, or within 6 BF16 ulps of the provider;
  - no GEMM row reports `interim`, and none of the three nodes drifts forward or backward.
- **`refiner-report.json`** (12, 257, 925 tokens): the same three rows and `attention` are `row` with the same classes, both bitwise properties hold, and the errors are within the bound.
- **Suites:**
  - head: 1 failed, 1474 passed, 81 skipped. The failure is the test above, the only one that fails only at head.
  - base: 1470 passed, 81 skipped.
- `check-h1.txt` has the per-layout check.

### H2. `h3-parallel-v2` (`<E1>`)

- **H2.a** (node-a, GPU 0):
  - `crosscheck.json` has `pass: true`, with 3 rows x 264 / 925 / 4296 rows x TP 1/2/4/8.
  - Every expected-bitwise piece is bitwise: the gate/up forward and `dW`, and the O and down `dX` and `dW`.
  - The reported-only pieces, at TP 8:
    - O forward: max abs 0.0156;
    - down forward: max abs up to 0.0313;
    - gate/up `dX`: max abs 0.0625.

    The differing counts are in the file.
  - `kernel-tests.txt`: 14 passed, none skipped.
- **H2.b** (2 x 4), all runs with `--sharded`, without `--provider-boundaries`:

| Run | Ranks | Result |
|---|---|---|
| `sharded_tp4_sp_928` | 4 | `all_bitwise: true`, `strict_boundaries: true` |
| `sharded_tp4_cp2_ring_925` | 8 | `all_bitwise: true`, `strict_boundaries: true` |
| `sharded_tp8_sp_928` | 8 | `all_bitwise: true`, `strict_boundaries: true` |
| `lora_random_tp4_cp2_sp_ulysses_928` | 8 | `all_bitwise: true`, `strict_boundaries: true` |
| `lora_zero_tp4_cp2_ring_925` | 8 | `all_bitwise: true`, `strict_boundaries: true` |
| `inject_sharded_tp4_cp2_ring_925` | 8 | first forward drift `block.17`, row 5 (1 row, text); first backward drift `block.49.grad`, row 77 |

- **Suites:** head 1525 passed, 87 skipped; base `81a87617` 1511 passed, 87 skipped. Nothing fails.

### H3. Strict G5 on the integrated head (`<E1>`, 2 x 4, TP4 x CP2 + SP)

None of the runs used `--provider-boundaries` or `--refiner-lora`. The peak is `grid_peak_bytes` (`torch.cuda.max_memory_allocated`) of each rank's report. It is the same on all 8 ranks.

| Run | Result | Peak per rank |
|---|---|---|
| `random_ring_928` | `pass: true`; `blocks_bitwise`, `refiner_bitwise`, both `repeat_*` true on all 8 ranks | 37.50 GiB |
| `random_ulysses_928` | same | 37.50 GiB |
| `zero_ring_928` | same | 37.50 GiB |
| `random_ring_928_clip` | same | 37.50 GiB |
| `inject_ring_928` | `pass: false`: only rank 2 fails (`reported_here`, 1 mismatch); `inject_detected: true` | 37.50 GiB |
| `random_ring_recipe` | `pass: true`, all flags true on all 8 ranks | 39.67 GiB (B200 with provider boundaries: 39.87 GiB) |

Every run has `strict_boundaries: true` and equal export digests. The grid's kernels include `H3AttentionOGemmOp`, `H3FFNGateUpGemmOp` and `H3FFNDownGemmOp` (`readback` in `report.json`).

### H4. #731 and #735 at their current heads (`<E2>`, node-b, one GPU)

- **#731 `f67d96d1`, step B:**
  - `verl-omni-compare.json` has 560 cases. `prev_sample`, `prev_sample_mean`, `std_dev_t` and `sqrt_dt` are 560 / 560, and `log_prob` is 206 / 560, with max abs 1.79e-07.
  - `logprob-batch.json` has `rl_kernel_differ: 0` (`verl_omni_differ: 33` of 144, as on H100 in the last round).
  - `transition-tests.txt`: 9 passed.
- **`transition-upstream.txt`:** 16 passed, including the three tests added at `f67d96d1`.
- **#735 `f3dbae37` with #734 (`8b391162`)**, with `RL_KERNEL_H3_WEIGHTS=<WEIGHTS>`:
  - `omni-cpu.txt`: 41 passed, 3 skipped, all 3 "CUDA required";
  - `omni-gpu.txt`: 44 passed.

### H5. Ring-attention block tests (`<E1>`, node-b, GPUs 0,1)

- `ring-attention.txt`: 2 passed, 8 skipped. The 8 skips need 3 to 8 GPUs, and the README's command makes two visible.
- The two that pass are the last round's failures, `cp2_ring_len261` and `cp2_ulysses_h3`. The supplementary `-v` run names them.
- `supplementary/` is not a README command: the same test with `-v` on 2 and on 4 GPUs. On 4 GPUs, 6 passed, and the 4 skipped need 6 to 8 GPUs.

## Deviations

1. **`<WEIGHTS>`** is `<SCRATCH>/h3ckpt`, the parent of `<CKPT>` = `<SCRATCH>/h3ckpt/hf_full`. It is the checkpoint of the last round: revision 42ed227e, with symlinks to the local `hf_full` plus a full download of `FL2VA`. For this round the directory also holds symlinks to the pinned `h3_block0.safetensors` and `h3_conditioning.safetensors` from the `prepare_h3_weights.py` output, so that one directory satisfies both `hf_full/` and the pinned files. The local `hf_full` beside the pinned files has only `FL2VA/audio_vae`.
2. **`test-h3` has moved.** Upstream `test-h3` is now `3f93cae8`, with #554 merged. The bundle's prerequisite `12e81c36` is its ancestor, and the bundle verifies. The results branch starts at `12e81c36`.
3. **H2.b harness relaunch.** In the first pass, the shell helper that records each step's state overwrote the variable that held the tool path. After the first run, the other five 8-rank runs therefore started `python` with the interpreter binary as the script, and exited with `SyntaxError` before any work. They are in `not-green-attempts/H2b-harness-launch-bug/` as they are. They were relaunched unchanged, with the same flags and new ports (`scripts/run_h2b.sh.txt`).
   - The first run, `sharded_tp4_sp_928`, launched correctly and is kept.
   - The same variable clash sent the first pass's exit codes to a stray file. They are split into `H2/rc-first-pass.txt` and `H3/rc.txt`, and `steps.txt` labels those runs `before <run>`.
   - H3 was not affected: it used a different variable for the tool path.
4. **#731 `f67d96d1` under `<E2>` has no `_C`.** Its build under torch 2.14 fails with `#error C++20 or later compatible compiler is required`; that branch does not have #734. The log is in `not-green-attempts/H4-731-build-E2/`. Step B and `test_h3_transition_upstream.py` ran without `_C`, as step B did at `b446400f` in the last round. `H4/c-file-731.txt` records the missing module.
5. **Launch.** The multi-rank tools ran under `srun`, one task per GPU, with rank 0 on node-a. `slurm_env.py` was not needed.
6. **Scheduling.** H1, then H2.a and the H2 suites, ran on node-a. At the same time, H4, then H5, ran on node-b. Each node ran one step at a time.
7. **Suite flags.** The suites ran with `--tb=line`, as the last round's suites did.

## Layout

- `H1` … `H5`: one folder per task, with `commit*.txt`, `env.txt`, `nvidia-smi*.txt`, `status-*.txt` (before and after), `c-file-*.txt` and `rc.txt`. `H2/<run>` and `H3/<run>` are the grid runs, each with `command.txt`, `run.log` and `report.json`.
- `build/`: `_C` build logs and checks.
- `code-check/`: bundle verification and the refs check.
- `check.txt`, `check-h1.txt`: the expectation checks (`scripts/check*.py.txt`).
- `scripts/`: the scripts as run.
- `steps.txt`: the step log.
- `not-green-attempts/`: the failed attempts, as they are.
