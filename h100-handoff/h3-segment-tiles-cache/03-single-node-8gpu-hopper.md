# 3. Single-node 8 x H100 runs (#569, #491, 10a NCCL cases)

**NOT PLANNED: the available hardware is two nodes x 4 H100.** Kept for anyone with a single 8-GPU node.

Three open PR notes ask for this, and the existing H100 evidence is from 4-GPU nodes only (#569 used two
nodes x 4 H100):
- #569: "the single-node 8-GPU Hopper request ... is still open"; `test_h3_tp_adaln.py` / `test_h3_sp_norm_adaln.py`
  not run on one 8-GPU Hopper node; the 8-rank TP8, TP4xCP2, TP4xCP2+SP layouts and the in-node hierarchical row.
- #491: H100 results have 2 and 4 GPUs, "No 8-GPU run".
- #587/#588 test tables: "the 8-GPU cases on H100" are skipped.
- #575 (`tp > 1`, adapter at tp = 8) has **no runner**: the report and tests use tp = 1. A tp = 8 harness would
  have to be written first, so it is not in this handoff.

**Requires one node with 8 H100 visible.** On a 4-GPU node every item below skips or cannot start; do not
substitute the two-node run, it is already published.

## A. #569 (branch `feat/h3-ws2-parallel`, head `718b0cad`)
Block-0 weights needed (`export RL_KERNEL_H3_WEIGHTS=<WEIGHTS>`). Strict build as usual.
```bash
( unset WORLD_SIZE RANK LOCAL_RANK CUDA_VISIBLE_DEVICES;
  python -m pytest -v -rs tests/distributed/sp/test_h3_sp_norm_adaln.py tests/distributed/tp/test_h3_tp_adaln.py -k nccl )
python -m pytest -q -rs tests/models/minimax_h3/test_h3_parallel.py
```
Layouts, one `srun` task per GPU (script docstring: `h3_parallel_run.py`); every one must report bitwise equal to
the single-GPU canonical block (forward, input gradient, weight-gradient shards):
```bash
srun -N1 --ntasks-per-node=2 python tools/validation/models/h3_parallel_run.py --tp 2 --out <OUT>/tp2
srun -N1 --ntasks-per-node=4 python tools/validation/models/h3_parallel_run.py --tp 4 --out <OUT>/tp4
srun -N1 --ntasks-per-node=8 python tools/validation/models/h3_parallel_run.py --tp 4 --cp 2 --out <OUT>/tp4_cp2
srun -N1 --ntasks-per-node=8 python tools/validation/models/h3_parallel_run.py --tp 4 --cp 2 --sp --out <OUT>/tp4_cp2_sp
srun -N1 --ntasks-per-node=8 python tools/validation/models/h3_parallel_run.py --tp 8 --local-size 8 --out <OUT>/tp8
srun -N1 --ntasks-per-node=8 python tools/validation/models/h3_parallel_run.py --tp 8 --local-size 4 --out <OUT>/tp8_hier_in_node
```
The last row selects the hierarchical IPC + NCCL tree inside one node; it is not a cross-node network
measurement and must be labelled that way. Charts: `tools/validation/models/h3_ws2_charts.py` (see its
`--help`). Send back each `report.json`, the pytest summary with skip reasons, and `nvidia-smi -q` header.

## B. #491 (branch `fix/deterministic-collective-fast-copy`, head `746c62b7`)
```bash
python -m pytest tests/distributed/test_deterministic_all_gather_sizes.py \
  tests/distributed/collectives/test_deterministic_all_gather.py -v
torchrun --standalone --nproc-per-node=8 benchmarks/benchmark_deterministic_collectives.py --output <OUT>/report8.json
```
Compare with the PR's own before/after method: the "before" build is the merge base plus this PR's benchmark
script, "after" the PR head, both back to back on the same node. Known: `test_deterministic_all_reduce.py` and
`test_deterministic_reduce_scatter.py` fail identically on unmodified main (misaligned address; `outs=`
argument); report them as "same as main" only if they are identical on the base build.

## C. Reporting
Per item: commit SHA, clean-tree check (`git status --porcelain` empty), driver/CUDA/torch versions, pass/skip
counts. Do not paste host names, job IDs, partitions or user paths into anything that will go on a PR.
