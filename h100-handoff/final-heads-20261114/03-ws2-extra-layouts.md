# 03. #701: the full-model layouts that were not run (TP4 x CP2 Ulysses, ring CP3)

Run this only after item 01 has passed for #701. Use the same `<SRC>/pr701` worktree and build.

On 2026-11-14 the full model was run with:
- Ulysses only at TP2 x CP2;
- ring at CP2 only.

Ring exists partly because Ulysses needs `heads % cp == 0`, and 56 heads rule out CP3. CP3 itself was never run.

```bash
export MASTER_ADDR=<first node>
run() { local name=$1; shift; WS2_NAME=$name srun --export=ALL "$@"; }
W="scripts/run_h100.sh ws2full <SRC>/pr701 <OUT>/03"
# Ulysses on the 2-node grid
run tp4-cp2-ulysses        -N2 --ntasks-per-node=4 $W --tp 4 --cp 2 --cp-mode ulysses --size 40
run tp4-cp2-ulysses-925    -N2 --ntasks-per-node=4 $W --tp 4 --cp 2 --cp-mode ulysses --size 925
run tp4-cp2-ulysses-sp-928 -N2 --ntasks-per-node=4 $W --tp 4 --cp 2 --cp-mode ulysses --sp --size 928
# ring CP3, one node, 3 GPUs (TP1)
run tp1-cp3-ring           -N1 --ntasks=3          $W --tp 1 --cp 3 --cp-mode ring --size 40
run tp1-cp3-ring-925       -N1 --ntasks=3          $W --tp 1 --cp 3 --cp-mode ring --size 925
# negative expectation: Ulysses must refuse CP3 (56 heads % 3 != 0)
run tp1-cp3-ulysses        -N1 --ntasks=3          $W --tp 1 --cp 3 --cp-mode ulysses --size 40
```

**Expected:**
- **The three Ulysses grids and the two ring CP3 grids:**
  - `all_bitwise = true`;
  - 350 weight shards compared and all bitwise on every rank;
  - `first_drift_*` null.

  CP3 at size 40 gives CP shards 14 / 13 / 13, and at size 925 gives 309 / 308 / 308: uneven by construction.
- **`tp1-cp3-ulysses`:** refused with `ValueError: 56 heads do not divide over cp=3`, from `ulysses_attention` in
  `parallel.py`. Record the message. Completing, or failing in any other way, is a finding.
- **A grid that refuses at set-up:** if the runner or `make_grid` refuses a layout before computing (for example TP1 or
  CP3 not supported somewhere), record the message and go on. That is a scope finding, not a failure to fix here.

**Why not TP2 x CP3 or TP4 x CP3:**
- TP4 x CP3 needs 12 GPUs.
- TP2 x CP3 needs 6 tasks on 2 nodes. With 3 + 3 tasks per node, one TP pair would span the nodes, but `tp <= local_size`
  selects the node-local IPC tree. Leave it out rather than depend on how `srun` places the tasks.

**Also out of scope:** SP together with uneven CP shards needs a size that the runner's fixed `SIZES` (12 / 40 / 925 /
928) does not offer, so it would need a runner change. Do not add one here.

## Send back

`03/<grid>/` (`report/report.json` and the per-rank folders; replace `host` with `node-a` / `node-b`), and a `SUMMARY.md`
with one row per grid:
- grid, size, CP shard sizes;
- `all_bitwise`, first drift;
- `weight_shards_compared`;
- load / canonical / parallel seconds and peak memory;
- the refusal message, where there is one.
