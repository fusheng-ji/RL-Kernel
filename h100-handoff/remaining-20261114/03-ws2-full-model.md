# 03. WS2 on the full model: ring CP, CP weight gradients, TP x CP x SP over 50 blocks

Branch `local/h3-ws2-fullmodel` @ `3f461dd9` (base `3baf93c8`, which is in `test-h3`). Four signed commits, never in a PR,
never run on H100. This is also RFC row `final_distributed_h3_run` (unclaimed): "multi-step TP4/CP2 run with sealed
per-step trajectory evidence". The branch gives one forward/backward step of it, not the multi-step part.

| Commit | What | Already in `test-h3` via #569? |
|---|---|---|
| `d6459f49` | ring context parallelism for the H3 attention kernels, bitwise equal to the unsharded op (`ring_attention.py`, `test_h3_ring_attention.py`, `deterministic_attention.cu`) | no: #569 has Ulysses only |
| `82ae6c08` | distributed packed IO: scatter from one rank, selected-row gather, row-identity check (`test_h3_packed_io.py`) | partly: #569 has `scatter_packed` / `gather_packed`, but not the selected-row gather or `check_row_identity` |
| `30d86afd` | CP weight gradients are the unsharded token sweep (exact gather, then the same fixed-order token GEMM); `run_tp_block_rows` | no: #569 "Limits" says the TP x CP block compares only forward and `dX` |
| `3f461dd9` | `h3_parallel_model.parallel_model_execution` (every block on `h3-parallel-v1`) and `tools/validation/models/h3_parallel_full_model.py` (canonical vs grid, `first_drift`, 350 weight-shard digests, `--inject` negative control) | no |

## Step 1. Merge-up and reconcile (no GPU)

```bash
git switch -c feat/h3-ws2-full-model local/h3-ws2-fullmodel
git merge --no-edit upstream/test-h3          # merge-tree: no textual conflict
```

Since `3baf93c8`, #569 changed four of the files this branch changes. Go through each one, because a clean textual merge
can still be wrong:
- `parallel.py`, `h3_tp.py`: `cca6e2fb` / `61619baf` refuse short CP shards in the compute paths. Ring CP and
  `cp_arbitrary_length` need shards of different lengths, down to 0 rows, so check that the ring path and
  `run_tp_block_rows` do not hit the new refusal, or that they hit it only where it is right.
- `parallel.py`: `9188133b` / `e2b09db7` create the hierarchical subgroups in one global order (`group_ranks`,
  `make_grid`). `h3_parallel_full_model.py` calls `make_grid(..., local_size=... if tp > local_size else None)`: check the
  CP > 1 cross-node case still passes `group_ranks`.
- `tests/models/minimax_h3/test_h3_parallel.py`: `0826d3d8` / `5ef8f21c` moved the WS2 harness import, and `6a00ba21` /
  `b1da8b64` skip the grids without weights. Keep both.
- `tools/validation/models/h3_parallel_run.py`: `cca6e2fb` corrected the WS2 timings and chart input.

Then deduplicate `82ae6c08` against #569's `scatter_packed` / `gather_packed`: keep one implementation and one test file.
Update `docs/operators/h3-parallel.md`: add ring CP, the CP weight gradients (rewrite the "Limits" line), and the full-model
runner.

## Step 2. Tests (one node, 4 GPUs)

```bash
git worktree add --detach <SRC>/ws2 feat/h3-ws2-full-model
scripts/run_h100.sh tests <SRC>/ws2 <OUT>/03/tests
scripts/run_h100.sh mypy  <SRC>/ws2 <OUT>/03/mypy
```

**Expected:** no failure beyond the `<SRC>/base` run of item 01. `test_h3_ring_attention.py`, `test_h3_packed_io.py`,
`test_h3_parallel_model.py` and the CP cases of `test_h3_parallel.py` (dW now compared) run, not skip.

## Step 3. Full-model grids (2 nodes x 4 H100)

Build once, then one `srun` per grid (one task per GPU). `h3_parallel_full_model.py` reads `SLURM_PROCID`,
`SLURM_NTASKS`, `SLURM_LOCALID` and `MASTER_ADDR`. The world size must be `tp * cp`. `--local-size 4` (GPUs per node) is
set by the script.

```bash
scripts/run_h100.sh build <SRC>/ws2 <OUT>/03/build
export MASTER_ADDR=<first node>
run() { local name=$1; shift; WS2_NAME=$name srun --export=ALL "$@"; }
G="scripts/run_h100.sh ws2full <SRC>/ws2 <OUT>/03"
run tp2              -N1 --ntasks=2                  $G --tp 2 --size 40
run tp4              -N1 --ntasks=4                  $G --tp 4 --size 40
run tp2-cp2-ring     -N1 --ntasks=4                  $G --tp 2 --cp 2 --cp-mode ring    --size 40
run tp2-cp2-ulysses  -N1 --ntasks=4                  $G --tp 2 --cp 2 --cp-mode ulysses --size 40
run tp4-cp2-ring     -N2 --ntasks-per-node=4         $G --tp 4 --cp 2 --cp-mode ring    --size 40
run tp4-cp2-ring-sp  -N2 --ntasks-per-node=4         $G --tp 4 --cp 2 --cp-mode ring --sp --size 40
run tp8              -N2 --ntasks-per-node=4         $G --tp 8 --size 40      # TP across the nodes: hierarchical collective
# negative control: BLOCK:GLOBAL_ROW:RANK. The row must lie inside that rank's span, or the injection is a silent no-op.
# Take ROW from "spans" in <OUT>/03/tp4-cp2-ring/report/report.json (rank 3's [lo, hi)).
run tp4-cp2-inject   -N2 --ntasks-per-node=4         $G --tp 4 --cp 2 --cp-mode ring --size 40 --inject 17:<ROW>:3
```

If `--size 40` passes on every grid, repeat `tp4-cp2-ring-sp` and `tp8` with `--size 925` (odd length, so the CP shards
are uneven) and `--size 928`.

**Memory risk.** Every rank loads the whole 33B checkpoint and runs the canonical single-GPU model before its grid
run. This has never run on an 80 GB H100. If it runs out of memory, record where (load / canonical / grid) and stop that
grid. Do not change the runner to offload or skip the canonical run: report it, and the fix belongs in a commit of its
own.

**Expected:**
- `report.json` has `all_bitwise = true` for every grid. Per rank, all `checks` are true:
  - video / audio velocity;
  - the three input gradients;
  - every block output and output gradient (`first_drift_*` is `null`);
  - `weight_shard_gradients_bitwise` over the 350 sharded projections.
- `readback` records the collective (`hierarchical_ipc_nccl_tree` for `tp8`), the rank order and the kernels.
- `tp4-cp2-inject`: `all_bitwise = false`, and `first_drift_forward` names block 17, the chosen row and rank 3 as its
  owner. With a row inside rank 3's span, any other location or no drift at all means the comparison is broken.

## Send back

`tests/`, `mypy/`, one `report.json` per grid (replace the `host` values with `node-a` / `node-b` before anything leaves the
machine), the per-rank logs, and `SUMMARY.md` with:
- one row per grid: grid, size, `all_bitwise`, first drift, load / canonical / parallel seconds, peak memory;
- the merge-up SHA and what step 1 found.
