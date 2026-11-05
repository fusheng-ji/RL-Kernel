# 1. Real-weight H3 suite at the three heads + hierarchical collective with CP > 1

One H100 node, 4 GPUs visible (`CUDA_VISIBLE_DEVICES=0,1,2,3`).

## 1a. Test suite

For each of `<SRC>/followup`, `<SRC>/pr569` and `<SRC>/pr575`, and once for the baseline `test-h3` at `cfec0d98`:

```bash
scripts/run_h100.sh tests <SRC>/<tree> <OUT>/01/<tree>
```

That runs `pytest tests/models/minimax_h3 -q -rs --junitxml=...` with the weights, the provider, vLLM-Omni and verl-omni
on the path, in the default build (`KERNEL_ALIGN_USE_FAST_MATH=0`).

**Expected:** no failure that the baseline does not also have. The tests that must now run with weights, not skip:
- `test_h3_one_block.py` and `test_h3_token_refiner.py` (follow-up and pr575): the block and refiner graphs now use the
  fused QKV backward. Their committed accuracy bounds against FP64 stay as they are;
- `test_h3_parallel.py::test_parallel_block_equals_single_gpu_bitwise[*]` (pr569, pr575), including the 4-GPU grids;
- `test_h3_checkpoint_mapper_vllm.py` (pr575), including the vLLM-Omni table comparison and
  `test_checkpoint_qkv_layout_is_what_the_vllm_omni_loader_reorders`;
- `test_h3_runtime_integration.py` (follow-up), including the verl-omni scheduler comparison.

**Send back:** passed / skipped / failed per tree and the baseline, the failures not in the baseline with their tracebacks,
and the skip reasons (so missing dependencies are visible).

On B200 (2026-11-05, 2 GPUs, no weights) the same trees had 0 failures, with 943 (follow-up), 972 (#569) and 1016 (#575)
passed. Those numbers are not comparable to a run with weights.

## 1b. Hierarchical collective with CP > 1 (OCR finding on #569, fixed in `e2b09db7`)

Before the fix, `HierarchicalCollective` created subgroups per TP group, so with CP > 1 different TP groups entered
`dist.new_group` with different rank lists. That could hang or build wrong groups. It now creates every TP group's
subgroups on every rank. Only a CPU simulation covers this so far. This is the first GPU run:

```bash
# one node, 4 tasks: two TP groups of 2 (CP 2), each group "spanning" 2 nodes of 1 GPU -> hierarchical path
cd <SRC>/pr569
srun -N1 --ntasks=4 --gpus-per-task=1 python tools/validation/models/h3_parallel_run.py \
    --tp 2 --cp 2 --local-size 1 --out <OUT>/01/hier-tp2-cp2-local1
# optional, two nodes x 4: two TP groups of 4, each split 2 x 2
srun -N2 --ntasks-per-node=4 python tools/validation/models/h3_parallel_run.py \
    --tp 4 --cp 2 --local-size 2 --out <OUT>/01/hier-tp4-cp2-local2
```

**Expected:** the run finishes (no hang at group creation). `report.json` shows `collective =
hierarchical_ipc_nccl_tree` and `all_bitwise = true` (forward and input gradient equal to the single-GPU block).

**Also worth one run:** the same command at the old #569 head `b1da8b64`, to see whether the old code hangs or mismatches.
It is diagnostic only: put a timeout on it (e.g. `timeout 900`) and record what happened.
