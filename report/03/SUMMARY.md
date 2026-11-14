# 03. WS2 on the full model (ring CP, CP dW, TP x CP x SP over 50 blocks)

Hardware: 2 nodes x 4 H100 (94 GB), torch 2.13.0+cu130, triton 3.7.1, provider `f53d5520`.
Hosts are `node-a` / `node-b`. Nothing was pushed.

## Result in one paragraph

`feat/h3-ws2-full-model` as handed over (`c1455a75`) passes its tests and mypy, but **none of its grids runs on a 94 GB
H100**: every rank runs out of memory in the canonical single-GPU run (the memory risk the item file warned about).
I stopped there, as asked, and made the fix in commits of their own on a new branch,
`fix/h3-ws2-full-model-memory` (`2fe3acb2`, on top of `c1455a75`). With that branch every grid is bitwise equal to
its canonical run, and the negative control is caught.

## Step 1. Merge-up and reconcile (no GPU)

- Merge-up SHA: `3d48548b` (`upstream/test-h3` into `feat/h3-ws2-full-model`, no textual conflict).
  Reconcile commit: `c1455a75` (amended once for isort).
- Findings:
  - **Stale import (would fail):** the ring, packed-IO and parallel-model tests imported `run_world` from
    `rl_engine.validation.models`; `test-h3` moved the WS2 harness to `rl_engine.validation.distributed.h3_ws2`. Fixed in
    the three test files.
  - **Short-shard refusal:** `cca6e2fb` / `61619baf` refuse a sequence shorter than `cp` in `run_tp_block`, and
    `seq_sharded_linear`. `run_tp_block_rows`, which the chained model enters, did not: with fewer rows than CP ranks,
    ranks would diverge. It now starts with `par.require_rows_on_every_rank(shard)`; test
    `test_chained_model_entry_refuses_a_sequence_shorter_than_cp_on_every_rank`. The ring and the packed IO themselves
    still accept ranks without rows (they do not go through this refusal).
  - **Hierarchical subgroups (`9188133b` / `e2b09db7`):** `h3_parallel_full_model.py` calls
    `make_grid(..., local_size=... if tp > local_size else None)` and `make_grid` passes `group_ranks`, so the TP
    subgroups are created in the one global order (`h3_tp.py`, `group_ranks=tp_ranks`). Run on 2 nodes: TP8 (cross-node
    TP, CP = 1) and TP4 x CP2 (CP across the nodes). The one combination that needs both, a TP group spanning nodes
    with CP > 1, needs 16 GPUs and was **not** run.
  - **`test_h3_parallel.py`:** the harness import of `0826d3d8` and the skip-without-weights of `6a00ba21` are both kept.
  - **`h3_parallel_run.py`:** `cca6e2fb`'s timing correction merged without change.
  - **Packed IO dedup (`82ae6c08`):** no duplicate. `#569` keeps `scatter_packed` / `gather_packed`; the branch adds
    `scatter_from_root`, `gather_selected` and `check_row_identity` next to them (different functions, same file), and
    `test_h3_packed_io.py` is distinct from `TestPackedIO` in `test_h3_parallel.py`. One implementation of each.
  - **Docs:** `docs/operators/h3-parallel.md` now covers ring CP, packed IO, CP dW (the "Limits" line is rewritten) and
    the full-model runner.

## Step 2. Tests and mypy (one node, 4 GPUs visible)

| Tree | SHA | passed | skipped | failed |
| --- | --- | --- | --- | --- |
| item 01 baseline (`base`) | `07533aee` | 1102 | 3 | 0 |
| `feat/h3-ws2-full-model` | `c1455a75` | 1179 | 18 | 0 |
| `fix/h3-ws2-full-model-memory` | `8db2cd98` | 1182 | 18 | 0 |
| `fix/h3-ws2-full-model-memory` | `2fe3acb2` | 1182 | 18 | 0 |

- Skips: the 3 of the baseline (two `aenum` for `vllm_omni`, one `sglang`), plus **15 that need 5-8 GPUs**
  (`test_h3_ring_attention.py`: 14, `test_h3_parallel_model.py`: 1). A node has 4 GPUs and the tests spawn local
  processes, so those 8-rank variants cannot run in pytest here. **Deviation from "run, not skip"**: the 8-rank ring /
  TP4xCP2 cases are covered by the two-node grids below instead.
- Ran, at `2fe3acb2`: `test_h3_ring_attention.py` 28 passed (14 skipped, see above), `test_h3_packed_io.py` 25 passed,
  `test_h3_parallel_model.py` 25 passed (1 skipped), `test_h3_parallel.py` 78 passed; the CP cases compare dW.
- mypy (`mypy --ignore-missing-imports rl_engine/` without torch, as CI): **clean** at `c1455a75` (667 files) and at
  `2fe3acb2`.
  With torch installed, `mypy rl_engine` reports 36 errors in 12 files at the baseline, and the same set at
  `2fe3acb2` (the pre-existing ones, in `rl_engine/integrations/...`). At `c1455a75` it reported 37: one more, in
  `ring_attention.py` (`torch.cat` of `list[None]`), fixed in `79fd9218` (annotation only).

## Step 3. Full-model grids at `c1455a75` (as handed over): all fail with CUDA out of memory

All 8 grids (`tp2`, `tp4`, `tp2-cp2-ring`, `tp2-cp2-ulysses`, `tp4-cp2-ring`, `tp4-cp2-ring-sp`, `tp8`, the
inject control; `--size 40`) fail the same way, on every rank: `torch.OutOfMemoryError` at
`h3_parallel_full_model.py:155` (the canonical `_run`), after the checkpoint loads (about 64 GB) and before any grid
run. Logs: `c1455a75/grids/<grid>/rank*/ws2.log`, GPU memory samples `c1455a75/grids/*.mem.csv` (about 95 GiB used).
The canonical run (`Grid(1,1)`) keeps every block's unsharded weight leaf and its gradient until the end, about twice
the 33B weights on top of the weights.

## Step 3. Full-model grids on `fix/h3-ws2-full-model-memory` (grids ran at `8db2cd98`)

The fix (own commits, the canonical run and the comparisons stay):

| Commit | What |
| --- | --- |
| `ad6b1a6d` | each weight-shard gradient is reduced to its digest in a post-accumulate hook and freed; the leaves are not kept. Compared quantities unchanged: digest of this rank's shard of the canonical gradient vs digest of the grid's shard gradient. |
| `8db2cd98` | a contiguous weight shard stays a **view** in the TP block leaves (it was cloned). Autograd holds each leaf until backward reaches it, so with a checkpointed forward the clones were all 50 blocks' weights a second time. **`ad6b1a6d` alone was not enough** (attempt 1 ran out of memory again in the canonical forward, at `h3_tp.py` `leaf()`, 91.3 GiB allocated; its rank logs were overwritten by the rerun). |
| `79fd9218`, `2b838164` | typing only (mypy with torch) |
| `2fe3acb2` | docs: results table and the memory note |

Every rank still loads the whole checkpoint and runs the canonical model; this is a validation harness, not
memory-sharded.

All rows: every check true on every rank (video / audio velocity, the three input gradients, every block output and
output gradient, `weight_shard_gradients_bitwise` over 350 sharded projections), `first_drift_*` null, 350 shards
compared, 0 mismatched. `readback`: `cuda_ipc_fixed_tree` (`DeterministicCollective`) within a node,
`hierarchical_ipc_nccl_tree` (`HierarchicalCollective`, 2 groups of 4) for TP8.

| grid | size | rows | GPUs | all_bitwise | first drift (fwd) | load s | canonical s | parallel s | peak GiB per node |
|---|---|---|---|---|---|---|---|---|---|
| tp2 | 40 | 40 | 2 | True | null | 69-70 | 10-11 | 1-1 | node-b:71.6 |
| tp4 | 40 | 40 | 4 | True | null | 76-77 | 10-10 | 2-2 | node-b:69.3 |
| tp2-cp2-ring | 40 | 40 | 4 | True | null | 62-76 | 10-11 | 2-3 | node-b:71.3 |
| tp2-cp2-ulysses | 40 | 40 | 4 | True | null | 66-72 | 10-11 | 3-3 | node-b:71.8 |
| tp4-cp2-ring | 40 | 40 | 8 | True | null | 53-78 | 10-11 | 2-2 | node-a:68.9 / node-b:68.1 |
| tp4-cp2-ring-sp | 40 | 40 | 8 | True | null | 54-74 | 10-10 | 2-2 | node-a:69.0 / node-b:68.1 |
| tp8 | 40 | 40 | 8 | True | null | 57-78 | 10-10 | 1-2 | node-a:67.8 / node-b:66.8 |
| tp4-cp2-ring-925 | 925 | 925 | 8 | True | null | 68-76 | 12-13 | 3-4 | node-a:69.5 / node-b:69.5 |
| tp8-925 | 925 | 925 | 8 | True | null | 54-77 | 12-13 | 3-4 | node-a:68.4 / node-b:67.2 |
| tp8-928 | 928 | 1008 | 8 | True | null | 53-77 | 10-11 | 3-4 | node-a:68.5 / node-b:68.5 |
| tp4-cp2-ring-sp-928 | 928 | 1008 | 8 | True | null | 55-79 | 10-11 | 5-5 | node-a:70.3 / node-b:69.4 |
| tp4-cp2-inject (control, no SP) | 40 | 40 | 8 | **False** | block 17 row 5 | 55-76 | 10-10 | 2-2 | node-a:68.2 / node-b:67.9 |

Notes:
- Seconds are the min-max over ranks. Peak memory is `nvidia-smi` `memory.used` sampled every 5 s (max over the
  node's GPUs), so a lower bound; all rows fit in 94 GB with 20+ GiB spare.
- Size `928` is a layout of **1008 rows**; `925` is 925 rows (CP shards 463 / 462).
- **`tp4-cp2-ring-sp` at size 925 is refused**, as designed: `ValueError: sequence parallelism needs the 463 rows of CP
  rank 0 divisible by tp=4 (length 925, cp=2)` (`tp4-cp2-ring-sp-925/`). As the uneven-CP-shard case I ran
  `tp4-cp2-ring` at 925 instead (row above).
- Negative control `--inject 17:5:3` on TP4 x CP2 (rank 3's span is rows `[0, 20)`; without SP the four TP ranks of a CP
  rank hold the same rows): `all_bitwise = false` on all 8 ranks, every check false;
  `first_drift_forward = {node: block.17, first_row: 5, rows_differing: 1, owner_ranks: [0, 1, 2, 3], modality: text}`.
  Block 17, row 5, and rank 3 is among the owners (the report lists all ranks that hold the row, not only the injector).
  **Owner rank:** without SP the four TP ranks of a CP rank hold the same rows, so this report cannot single out the
  injector. A second control with SP (`tp4-cp2-ring-sp-inject`, `--sp --inject 17:16:3`; rank 3's span is rows
  `[15, 20)`) gives `all_bitwise = false` and `first_drift_forward = {node: block.17, first_row: 16, rows_differing: 1,
  owner_ranks: [3], modality: audio}`: block 17, the chosen row, and **rank 3 as the only owner**.
  The first backward drift is reported at `block.49.grad` (audio, row 8), because the backward walk starts at the end of
  the model. 10 weight shards are listed as mismatched (the report truncates the list at 10).
- Per-rank `report.json` files have `host` replaced by `node-a` / `node-b`.

## What needs a decision

- The grids only run with the three memory-fix commits (`ad6b1a6d`, `8db2cd98`, and the typing fixes). They belong in the
  WS2 PR (or in a PR before it); `8db2cd98` touches `rl_engine/validation/models/h3_tp.py` (leaf is a view).
- 15 pytest cases need 5-8 GPUs on one node and are skipped here.
- Not done: the multi-step part of `final_distributed_h3_run` (this is one forward / backward step).

## Why each memory-fix / session commit is needed, and what covers it

| Commit | Why | Covered by |
| --- | --- | --- |
| `a6c93313` (item 01 branch) | The strict video projection rows (#562 / #646) refuse parameter gradients without a `canonical_backward_session`, `logical_keys` and a `parameter_id`; without it every strict full-model run fails (`fulltrans` on `test-h3`, and the WS2 canonical run). | Unit: `test_h3_full_model.py` strict-boundary routing test (keys, `parameter_id`, open session for `proj_in` / `proj_out`; none for the others; session closed on exit). Real weights: strict `fulltrans` completes. |
| `ad6b1a6d` | The canonical run kept every block's unsharded leaf and gradient (about 2x the 33B weights on top of the weights). Digests are taken in a post-accumulate hook and the gradient is freed. | Unit: `test_h3_parallel_model.py::TestDigestOnGrad` (3 CPU tests). End to end: only the 33B grids. Not sufficient alone (first tp2 retry still ran out of memory). |
| `8db2cd98` | Autograd holds each leaf until backward reaches it, so the cloned leaves of a checkpointed forward were all 50 blocks' weights again. A contiguous shard is now a view. | **No unit test.** Only the 33B grids (all out of memory before, all fit after). A small CPU test (leaf shares storage with a contiguous weight) would close the gap; not written. |
