# 1. Backward segment-tile cache: H100 evidence

Local commit `6bd3fe74` (patch `0001-segment-tiles-cache.patch`, not pushed, no PR). It builds the
sorted-segment tile metadata of the deterministic gate/shift/scale gradients once per row index
instead of once per norm / gated residual. Kernels unchanged; gradients must stay bitwise equal.

Already measured elsewhere (RTX PRO 6000 Blackwell, shared and loaded node, so the timings are not
usable): new tests 6/6 pass; 331 passed / 66 skipped on 1 GPU, 159 passed on 2 GPUs for the
two-device validation subset; 50 chained candidate blocks (random weights) build the metadata 200
times without the cache and once with it, input gradient bitwise equal.

Open on H100: (a) the 66 skipped tests (they need the block-0 weights, 2 or more GPUs, 4/8 GPUs);
(b) timings on an idle card; (c) whether the **real** 50-layer model hands every block the same
`adaln_indices` tensor (the one point not verified: the model class is not available where the
cache was written). If it does not, the cache only helps inside a block (4 builds -> 1) not across
blocks.

## Setup
```bash
git clone <fork> <REPO> && cd <REPO>
git fetch origin feat/h3-generation-gate        # head 67365734 = #588, has tools/.../h3_full_model.py
git worktree add ../tc-base  a88b5a09           # before, plain upstream test-h3
git worktree add ../tc-cache a88b5a09 && (cd ../tc-cache && git am ../<this dir>/0001-segment-tiles-cache.patch)
git worktree add ../tc-588   67365734
git worktree add ../tc-588c  67365734 && (cd ../tc-588c && git am ../<this dir>/0001-segment-tiles-cache.patch)
# build once per tree with ninja on PATH (strict H3 build); the kernels are the same in all four
( cd ../tc-cache && RL_KERNEL_H3_STRICT=1 python setup.py build_ext --inplace )
# reuse the .so in the others:  cp ../tc-cache/rl_engine/_C*.so ../tc-base/rl_engine/ (etc.)
```
Check `python -c "import rl_engine._C as c; print(c.__file__)"` inside each tree points to that tree.
If weights are missing: `python tools/weights/prepare_h3_weights.py --out <WEIGHTS> --block`, then
`export RL_KERNEL_H3_WEIGHTS=<WEIGHTS>`.

## A. Tests (tc-cache, 2 GPUs visible, weights set)
```bash
python -m pytest -q -rs -p no:cacheprovider \
  tests/models/minimax_h3/test_h3_adaln_row_gather.py tests/models/minimax_h3/test_h3_adaln_gate_residual.py \
  tests/models/minimax_h3/test_h3_adaln_modulation.py tests/models/minimax_h3/test_h3_final_adaln_out.py \
  tests/models/minimax_h3/test_h3_rmsnorm.py tests/models/minimax_h3/test_h3_rmsnorm_native_validation.py \
  tests/models/minimax_h3/test_h3_native_validation.py tests/distributed/sp/test_h3_sp_norm_adaln.py
```
Expect 0 failures. Send back the pass/skip counts and the `-rs` skip reasons (4/8-GPU cases will
still skip on a 2-GPU box). Run the same on tc-base and report any test that differs.

## B. Timings on an idle GPU (one card, nothing else running; `nvidia-smi` shows 0 %)
```bash
for t in base cache base cache base cache; do
  (cd ../tc-$t && CUDA_VISIBLE_DEVICES=0 python <this dir>/scripts/bench_tiles.py | tail -3)
done
(cd ../tc-cache && python <this dir>/scripts/probe_50_blocks.py small)   # needs ~25 GiB; also try: tiny
```
`bench_tiles.py` prints `_segment_tiles`, gate_residual fwd+bwd and modulated rmsnorm fwd+bwd at
S=4097, H=5376. Alternate base/cache and report the min and median of each, plus `nvidia-smi -q`.
`probe_50_blocks.py` must print `input grad bitwise equal cached vs uncached: True`, 1 build with the
cache and 200 without; the fwd/bwd ms are indicative only.

## C. Real model: does the cache hit across the 50 blocks? (tc-588c, 1–2 H100, checkpoint)
```bash
python <this dir>/scripts/count_tile_builds_full_model.py --checkpoint <CKPT> \
  --manifest rl_engine/validation/models/h3_full_manifest.json --mode candidate \
  --thresholds examples/models/minimax_h3/full-model-thresholds.json --output <OUT>/cached
H3_TILES_NO_CACHE=1 python <this dir>/scripts/count_tile_builds_full_model.py ... --output <OUT>/uncached
```
Send back both `tile_builds.json` files. Reading: uncached, `tile_builds` equals `consumer_lookups`
(every lookup rebuilds). Cached, `tile_builds` far below `consumer_lookups` and near 1 means the model
hands every block the same index (cross-block hits confirmed); near the number of blocks (about 50)
means a fresh index per block, so only the within-block saving applies. Also keep the two
`report.json` files the runner writes (step times) and confirm the 638 parameter-gradient
comparisons are identical between the two runs (they must be: same kernels, same tiles).
