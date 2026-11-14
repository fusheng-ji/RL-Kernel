# 01. The final heads on H100: #699 `0566cdbb`, #700 `3eda7d4f`, #701 `e42c763a`

Default build (`KERNEL_ALIGN_USE_FAST_MATH=0`). Set `PY`, `CKPT`, `WEIGHTS`, `PROVIDER`, `VERL_OMNI` first.

## #699 (one node, 4 GPUs visible for the suite, then one GPU per run)

```bash
G=scripts/run_h100.sh
$G tests     <SRC>/pr699 <OUT>/01/pr699/tests
$G mypy      <SRC>/pr699 <OUT>/01/pr699/mypy
$G fulltrans <SRC>/pr699 <OUT>/01/pr699/fulltrans      # strict, then --provider-boundaries
$G dual      <SRC>/pr699 <OUT>/01/pr699/dual           # seeds 11 22 33
```

**Expected:**

| Run | Last H100 result |
|---|---|
| suite | 1115 / 3 / 0: 1114 at `e3c4cbc9`, plus the 1 test added since (counted by collection). The skips are two `aenum` and one `sglang` |
| strict `fulltrans` | completes; candidate and provider both bitwise repeatable |
| `--provider-boundaries` | 2.211e-03 / 2.053e-02 / 0.142 / 0.146 / 0.247 / 0.282 / 0.924; the path is unchanged, so these columns are expected **identical** |
| `dual` | strict 0/16 and provider 16/16 on every seed |

- In strict mode, `boundary_bindings` has six `row` and two `provider` entries (`time_embedder`, `norm_out`).
- The strict numbers were video 2.151e-03, audio 2.104e-02, input grads 0.226 / 0.230 / 0.345, param grads
  0.266 / 0.696. `0566cdbb` changes only the error path, so these are expected **identical**. Any difference is a
  finding.
- `dual` runs without gradients, so it is unaffected by `0566cdbb`; it is the check that nothing else moved.

## #700 (same node)

```bash
$G tests     <SRC>/pr700 <OUT>/01/pr700/tests
$G mypy      <SRC>/pr700 <OUT>/01/pr700/mypy
$G fulltrans <SRC>/pr700 <OUT>/01/pr700/fulltrans
$G dual      <SRC>/pr700 <OUT>/01/pr700/dual
```

**Expected:**

| Run | Expected result |
|---|---|
| suite | 1122 / 3 / 0: 1118 at `89cbd77f`, plus the 4 tests added since (counted by collection) |
| strict `fulltrans` | `boundary_bindings`: **all eight `row`**; candidate bitwise repeatable; numbers **identical** to `89cbd77f`: video 2.511e-03, audio 1.543e-02, input grads 0.118 / 0.156 / 0.248, param grads 0.176 / 0.434 |
| `--provider-boundaries` | identical to #699's |
| `dual` | strict 0/16, provider 16/16 |

- **The first thing to check in this item:** if strict `fulltrans` raises
  `strict timestep_mlp_fp32 resolved to ..., not its CUDA op` (or the same for `final_adaln_out`), that is a real
  finding in `8557fef6`.
  - Record the class it resolved to.
  - Record the CUDA op's construction error: run `KernelRegistry().get_op("timestep_mlp_fp32", device="cuda")` and
    read the logged `Failed to instantiate ...`.
  - Then stop #700 and report.

## #701 (2 nodes x 4 H100 for the grids)

```bash
$G tests <SRC>/pr701 <OUT>/01/pr701/tests                # one node, 4 GPUs
$G mypy  <SRC>/pr701 <OUT>/01/pr701/mypy
$G build <SRC>/pr701 <OUT>/01/pr701/build                # build once before the grids
export MASTER_ADDR=<first node>
run() { local name=$1; shift; WS2_NAME=$name srun --export=ALL "$@"; }
W="scripts/run_h100.sh ws2full <SRC>/pr701 <OUT>/01/pr701"
run tp4-cp2-ring           -N2 --ntasks-per-node=4 $W --tp 4 --cp 2 --cp-mode ring --size 40
run tp8-928                -N2 --ntasks-per-node=4 $W --tp 8 --size 928
run tp4-cp2-ring-sp-inject -N2 --ntasks-per-node=4 $W --tp 4 --cp 2 --cp-mode ring --sp --size 40 --inject 17:16:3
```

**Expected:**
- **Suite:** no failure. At `2fe3acb2` it was 1182 / 18 / 0, and this head adds 5 CPU tests (`TestShardLeaf`), so
  1187 / 18 / 0. The 15 extra skips are the 5–8-GPU cases.
- **`tp4-cp2-ring`, `tp8-928`:**
  - `all_bitwise = true`;
  - `weight_shards_compared = 350` and `weight_shard_gradients_bitwise = true` on every rank;
  - `first_drift_*` is null.

  `e42c763a` made the shard check stricter, and these grids are the check that it still passes on a real run.
- **`tp4-cp2-ring-sp-inject`:** `all_bitwise = false`, and
  `first_drift_forward = {node: block.17, first_row: 16, owner_ranks: [3]}` on every rank. Rank 3's span is
  `[15, 20)` at size 40 with SP.

## Send back

`01/pr699/`, `01/pr700/` and `01/pr701/`, plus one `SUMMARY.md` with:
- **Tests:** passed / skipped / failed for each head. For a failure, the traceback and whether `test-h3` has it too.
- **Full transformer:** the strict and provider-boundaries tables for #699 and #700 with this run's numbers, and the
  difference against the numbers above. They are expected to be zero.
- **Dual trajectory:** the table for #699 and #700.
- **#701 grids:** one row per grid: `all_bitwise`, first drift, `weight_shards_compared`, peak memory.
