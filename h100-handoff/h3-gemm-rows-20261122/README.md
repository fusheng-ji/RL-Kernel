# H100 handoff: H3 GEMM rows (#535, #554) in the strict model and the TP path, and current-head reruns (2026-11-22)

## Why

The O, gate/up and down projections now run their RFC rows' operators:
- `H3AttentionOGemmOp` and `H3FFNGateUpGemmOp` from #554;
- `H3FFNDownGemmOp` from #535.

Those operators run on **SM90 only**, so the WS1 evidence below can only come from Hopper. B200 covers the TP grids (`h3-parallel-v2`), G5 and the multi-step G6 run, with the rows' kernel at shard shapes.

The new code is in three local branches and an integrated head (PR E). None of them is a PR yet, so they come in the bundle. The current heads of #731 and #735 also changed after the last H100 round, so H4 reruns their tests at those heads.

| Task | Code | Hardware | Environment | Time (est.) |
|---|---|---|---|---|
| **H1** WS1 GEMM rows | `h100/h3-strict-gemm-rows` | 1 node, 1 + (2 + 2) GPUs | `<E1>` | about 1 h |
| **H2** TP v2 against the rows' operators | `h100/h3-parallel-v2-gemm-rows` | 1 GPU, then 2 nodes x 4 | `<E1>` | about 1.5 h |
| **H3** strict G5 on the integrated head | `h100/h3-integrated` | 2 nodes x 4 H100 | `<E1>` | about 1 h |
| **H4** #731 and #735 at their current heads | #731 `f67d96d1`; `h100/pr735-f3dbae37-with-734` | 1 GPU | `<E2>` | about 40 min |
| **H5** the two S1 suite tests that needed weights | #733 `0b5b87c3` | 1 node, 2 GPUs | `<E1>` | about 10 min |

Paths are placeholders, as in the earlier rounds:
- `<REPO>`: a clone of the fork;
- `<THIS_DIR>`: this folder;
- `<OUT>`: where results are written;
- `<CKPT>`: the H3 checkpoint, revision `42ed227ee7df40d41602854ae760620d6eb651fe`;
- `<WEIGHTS>`: the directory that holds `hf_full/` (the parent of `<CKPT>`, as found in the last round);
- `<E1>`, `<E2>`, `<PROVIDER>`, `<VERL_OMNI>`.

`<E1>` and `<E2>` are the environments of the last round (`results/pip-list-filtered-E*.txt` on `handoff/h100-h3-hopper-strict-results-20261122`).

## Rules (unchanged from the earlier H3 rounds)

- **Clean trees.** Use detached worktrees. `git status --porcelain --untracked-files=no` must be empty before and after every step. Change no file.
- **Check `_C`.** Before every GPU step, check that `rl_engine._C.__file__` is inside the checkout.
- **Idle hardware.** Use idle nodes with nothing else on the GPUs, and record `nvidia-smi` at the start of each step.
- **Report failures as they are.** A failure is reported as a failure, with its log. Never change a script, a test or a bound to make a result pass. Failed attempts go, as they are, into `not-green-attempts/`.
- **Push only the results branch.** No PR, no comment.
- **Redact everything sent back.**
  - Remove: cluster, job, node, partition, user, conda environment name, `/home/…` or `/data/…` path, email. Hosts become `node-a` / `node-b`.
  - `git add -f` the `*.log` files.
  - Commit with `git commit -s`, with no AI attribution lines.

## Code

```bash
cd <REPO>
git fetch https://github.com/fusheng-ji/RL-Kernel.git handoff/h100-h3-gemm-rows-20261122 \
  feat/h3-final-run-transition-norm feat/h3-omni-rollout-model feat/h3-parallel-lora \
  merge/pr554-test-h3 build/cxx20-torch214
git fetch https://github.com/RL-Align/RL-Kernel.git test-h3
git checkout FETCH_HEAD -- h100-handoff/h3-gemm-rows-20261122   # or read it from the branch
git bundle verify <THIS_DIR>/h3-gemm-rows-20261122.bundle
git fetch <THIS_DIR>/h3-gemm-rows-20261122.bundle 'refs/h100/*:refs/h100/*'
```

`refs.txt` lists every ref with its commit. The bundle's prerequisites are the fork branches fetched above and `test-h3` `12e81c36`.

| Ref | Commit | What |
|---|---|---|
| `refs/h100/h3-strict-gemm-rows` | `775e98e7` | PR 1: the block and refiner bind the three GEMM rows (`CandidateOps(gemm_rows=...)`); on #554's merge with `test-h3` (`9388feed`) |
| `refs/h100/h3-parallel-v2-gemm-rows` | `093b0ff2` | PR 2: `h3-parallel-v2`, TP gate/up / O / down on the rows' kernel, plus the cross-check tool; on #733 `0b5b87c3` + `9388feed` |
| `refs/h100/h3-pack-noise-rows` | `2c9adb7e` | PR 3: `packed_index_copy` and `conditioning_noise_mix` rows (not run here; CPU-tested) |
| `refs/h100/h3-integrated` | `b6e798c8` | PR E (grid training step) with #731, #735 and PR 1-3 merged, plus the #734 build fix |
| `refs/h100/pr735-f3dbae37-with-734` | `8b391162` | #735's current head with #734, for `<E2>` (torch 2.14) |

Build `_C` in each checkout:

```bash
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E>/bin/python setup.py build_ext --inplace
```

## Launching multi-rank tools

`h3_parallel_full_model.py` and `h3_grid_train_step.py` read the SLURM rank variables, so launch them as in the last round:
- under `srun` with one task per GPU, run them directly;
- under `torchrun`, go through `<THIS_DIR>/slurm_env.py`.

For 8 ranks on 2 nodes, global rank 0 is on node-a. Use a new port for every run.

---

## H1. WS1 GEMM rows (`<E1>`, `h100/h3-strict-gemm-rows`)

```bash
git worktree add --detach <W1> refs/h100/h3-strict-gemm-rows
git worktree add --detach <B1> 9388feed     # base: #554 merged with test-h3 (origin/merge/pr554-test-h3)
cd <W1>; export PYTHONPATH=<W1>:<PROVIDER>/src RL_KERNEL_H3_WEIGHTS=<WEIGHTS>
CUDA_VISIBLE_DEVICES=0 <E1>/bin/python -m pytest -q -p no:cacheprovider -rfs \
  tests/models/minimax_h3/test_h3_one_block.py tests/models/minimax_h3/test_h3_token_refiner.py \
  tests/models/minimax_h3/test_h3_operator_trace.py tests/models/minimax_h3/test_h3_full_model.py \
  > <OUT>/H1/binding-tests.txt 2>&1
CUDA_VISIBLE_DEVICES=0 <E1>/bin/python tools/validation/models/h3_block_replay.py \
  --layouts tiny,small,medium --out <OUT>/H1/block-replay.json > <OUT>/H1/block-replay.log 2>&1
CUDA_VISIBLE_DEVICES=0 <E1>/bin/python tools/validation/models/h3_refiner_report.py \
  --tokens 12 257 925 --out <OUT>/H1/refiner-report.json > <OUT>/H1/refiner-report.log 2>&1
```

Then run the suite at head and at base, `tests/models/minimax_h3`, on 2 + 2 GPUs with `RL_KERNEL_H3_WEIGHTS=<WEIGHTS>` set on both sides. Write the output to `<OUT>/H1/suite-head.txt` and `suite-base.txt`.

**Expected:**
- **Binding tests.** `test_gemm_rows_run_their_operators_on_sm90` runs, is not skipped, and passes. All other tests pass.
- **`block-replay.json`.** In every layout's `nodes`:
  - `o_proj`, `ffn_gate_up` and `ffn_down` have `status: row` and the rows' operator classes as `backend`;
  - `attention` is `row` (`H3FullAttentionOp`);
  - `repeat_bitwise_equal` and `batch_row_bitwise_equal` hold for every node, forward and backward;
  - each node's `candidate_rel_err_vs_golden` is within the tests' bound of the provider's.
- **`refiner-report.json`.** The same three rows are `row`.
- **Suite.** Nothing fails only at head.

A GEMM row that reports `interim` on H100, or a forward or backward drift at one of the three nodes, is the main finding: send it as is.

## H2. `h3-parallel-v2` against the rows' operators (`<E1>`, `h100/h3-parallel-v2-gemm-rows`)

### H2.a One GPU

```bash
git worktree add --detach <W2> refs/h100/h3-parallel-v2-gemm-rows
cd <W2>; export PYTHONPATH=<W2>:<PROVIDER>/src RL_KERNEL_H3_WEIGHTS=<WEIGHTS>
CUDA_VISIBLE_DEVICES=0 <E1>/bin/python tools/validation/models/h3_gemm_rows_crosscheck.py \
  --rows 264 925 4296 --out <OUT>/H2/crosscheck.json > <OUT>/H2/crosscheck.log 2>&1
CUDA_VISIBLE_DEVICES=0 <E1>/bin/python -m pytest -q -p no:cacheprovider -rfs \
  tests/models/minimax_h3/test_h3_fixed_fp32_gemm.py > <OUT>/H2/kernel-tests.txt 2>&1
```

**Expected:** `crosscheck.json` `pass: true`, which covers three rows × three row counts × TP 1/2/4/8:
- **bitwise**, against the rows' TP=1 operators run with logical keys in a canonical session:
  - gate/up forward columns;
  - O and down `dX` columns;
  - every `dW` shard;
- **reported only, not bitwise:** the O and down forward and the gate/up `dX`. These are the 8-unit tree, so the file records their max abs and differing counts.

`kernel-tests.txt`: everything passes, including the SM90 cross-check test.

### H2.b Strict-boundary grids at v2 (2 nodes x 4, `h3_parallel_full_model.py`)

These are S1's runs at the new head. Use `COMMON="--checkpoint <CKPT> --attention h3_full --out <OUT>/H2/<run>"`, **no** `--provider-boundaries`, and `--sharded` on every run.

| Run | Ranks | Flags (after `$COMMON --sharded`) |
|---|---|---|
| `sharded_tp4_sp_928` | 4 | `--tp 4 --sp --size 928` |
| `sharded_tp4_cp2_ring_925` | 8 | `--tp 4 --cp 2 --cp-mode ring --size 925` |
| `sharded_tp8_sp_928` | 8 | `--tp 8 --sp --size 928 --local-size 4` |
| `lora_random_tp4_cp2_sp_ulysses_928` | 8 | `--tp 4 --cp 2 --sp --cp-mode ulysses --size 928 --lora-rank 64 --lora-alpha 128 --lora-seed 7` |
| `lora_zero_tp4_cp2_ring_925` | 8 | `--tp 4 --cp 2 --cp-mode ring --size 925 --lora-rank 64 --lora-alpha 128 --lora-seed 7 --lora-b-zero` |
| `inject_sharded_tp4_cp2_ring_925` | 8 | `--tp 4 --cp 2 --cp-mode ring --size 925 --inject 17:5:3` |

Then run the suite at head and at base, `tests/models/minimax_h3`, on 2 + 2 GPUs, with the base at `81a87617` (in the bundle: #733 `0b5b87c3` merged with `9388feed`). Write the output to `<OUT>/H2/suite-*.txt`.

**Expected:**
- every non-inject run is `all_bitwise: true` with `strict_boundaries: true`;
- the inject run drifts first at `block.17`, row 5;
- nothing fails only at head.

The bytes differ from S1, because the kernel changed; only the equalities are compared.

## H3. Strict G5 on the integrated head (2 nodes x 4, `<E1>`, `h100/h3-integrated`)

```bash
git worktree add --detach <W3> refs/h100/h3-integrated
cd <W3>; export PYTHONPATH=<W3>:<PROVIDER>/src
COMMON="--checkpoint <CKPT> --tp 4 --cp 2 --sp --repeat --passes 3 --optimizer-steps 2"
```

| Run | Flags (after `$COMMON`) |
|---|---|
| `random_ring_928` | `--cp-mode ring --size 928 --out <OUT>/H3/random_ring_928` |
| `random_ulysses_928` | `--cp-mode ulysses --size 928 --out ...` |
| `zero_ring_928` | `--cp-mode ring --size 928 --lora-b-zero --out ...` |
| `random_ring_928_clip` | `--cp-mode ring --size 928 --max-norm 1e-3 --out ...` |
| `inject_ring_928` | `--cp-mode ring --size 928 --max-norm 1e9 --inject 17.attn.to_v.A:2 --out ...` |
| `random_ring_recipe` | `--cp-mode ring --size recipe --out ...` |

Each command is `tools/validation/models/h3_grid_train_step.py $COMMON <flags>`, with 8 ranks.

These runs deliberately omit two flags that B200 uses:
- **no `--provider-boundaries`**: every boundary module is strict, which only Hopper can run;
- **no `--refiner-lora`**: the strict refiner row refuses a refiner LoRA by design.

On SM90 the canonical and grid refiners both run the new GEMM rows' operators.

**Expected:**
- every non-inject run prints `"pass": true`, with `blocks_bitwise`, `refiner_bitwise` and both `repeat_*` true on all 8 ranks;
- `inject_ring_928` fails on rank 2 only.

Record each run's per-rank peak memory from the log. The B200 reference, with provider boundaries, is 39.87 GiB at the recipe size.

## H4. #731 and #735 at their current heads (`<E2>`, one GPU)

**#731 `f67d96d1`.** It adds an optional `noise=` argument to the drop-in transition; the default path is unchanged. Run step B of `handoff/h100-h3-final-run-pra-20261120` exactly as written there, with `<SRC>` at `f67d96d1`. Also run:

```bash
CUDA_VISIBLE_DEVICES=0 VERL_OMNI_PATH=<VERL_OMNI> <E2>/bin/python -m pytest -q -p no:cacheprovider -rfs \
  tests/models/minimax_h3/test_h3_transition_upstream.py > <OUT>/H4/transition-upstream.txt 2>&1
```

**Expected** (as at `b446400f` in the last round):
- `prev_sample`, `prev_sample_mean`, `std_dev_t` and `sqrt_dt` are 560 / 560;
- `log_prob` is 206 / 560;
- `rl_kernel_differ` is 0;
- every test in `transition-upstream.txt` passes, including the three tests added at `f67d96d1`.

**#735 `f3dbae37`**, as `refs/h100/pr735-f3dbae37-with-734`. It adds the G3 manifest checks and the worker's failure paths. Run `tests/models/minimax_h3/test_h3_omni_model.py` with `RL_KERNEL_H3_WEIGHTS=<WEIGHTS>`, once with `CUDA_VISIBLE_DEVICES=""` and once on one GPU. Write the output to `<OUT>/H4/omni-cpu.txt` and `omni-gpu.txt`.

**Expected:**
- on CPU, every test passes except the CUDA-only ones, which skip;
- on GPU, every test passes.

## H5. The two S1 suite tests that needed weights (`<E1>`, #733 `0b5b87c3`, 2 GPUs)

```bash
RL_KERNEL_H3_WEIGHTS=<WEIGHTS> CUDA_VISIBLE_DEVICES=0,1 <E1>/bin/python -m pytest -q -p no:cacheprovider -rfs \
  "tests/models/minimax_h3/test_h3_ring_attention.py::test_block_with_h3_attention_equals_single_gpu_bitwise" \
  > <OUT>/H5/ring-attention.txt 2>&1
```

**Expected:** every parametrization passes. The last round's two failures were only the unset weights path.

## What to send back

Push one branch, `handoff/h100-h3-gemm-rows-results-20261122`, with:
- `results/H1` … `results/H5`;
- `results/SUMMARY.md`, with one verdict per task and every deviation;
- `results/pip-list-filtered-E1.txt` and `-E2.txt`.

Each step folder has `commit.txt`, `env.txt`, `nvidia-smi*.txt`, and the tree status before and after.
