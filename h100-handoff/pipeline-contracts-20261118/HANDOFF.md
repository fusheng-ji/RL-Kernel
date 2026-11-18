# H100 handoff: FL2VA / Ref2VA pipeline contracts (RFC #420 Phase B)

Nothing in the two rows' own contracts needs Hopper. Preprocessing and packed-layout construction are CPU arithmetic, and the CUDA transfer of the layout and of the initial rows is byte-equal on B200 (tested).

## Code

This branch, `handoff/h3-h100-pipeline-contracts-20261118`, is `feat/h3-fl2va-pipeline-contract` @ `4dfd70df` merged with `feat/h3-ref2va-pipeline-contract` @ `75778921` (both on base `10552d82`, `test-h3`), plus one commit that adds this folder. `<worktree>` below is a checkout of it:

```bash
git fetch https://github.com/fusheng-ji/RL-Kernel handoff/h3-h100-pipeline-contracts-20261118
git worktree add --detach <worktree> FETCH_HEAD
```

**What can run now:** item 1 and item 3. Item 2 also needs `qwen3vl_layer50_contract` and `phase_b_full_chain`, which are still in progress; they will get their own handoff. The visual and audio VAE rows have their own handoff branches (`handoff/h3-h100-visual-vae-20261118`, `handoff/h3-h100-audio-vae-20261118`).

The items below are what the RFC still asks of these rows beyond this PR. Each needs the strict full model, which is Hopper-only (`#646` video rows need cc 9.0), the `transformer_ref/` weights, or both.

## 1. Identify both transformer partitions (cheap, no GPU)

`transformer_ref/` has never been downloaded on our side: the B200 checkpoint dir holds `transformer/` only. On the H100 node, after the pinned download (`MiniMaxAI/MiniMax-H3@42ed227`, `hf download --local-dir`):

```bash
PYTHONPATH=<worktree> python -c "
from rl_engine.models.minimax_h3 import packed_layout as pl
for d in ('transformer', 'transformer_ref'):
    print(d, pl.identify_transformer_dir('<ckpt>/' + d))                 # Hub metadata, instant
    print(d, pl.identify_transformer_dir('<ckpt>/' + d, full_hash=True))  # reads ~66 GB each
"
```

Expected: `transformer FL2VA`, `transformer_ref Ref2VA`, both ways. A `VariantMismatch` means a shard is from the other partition.

## 2. One sealed end-to-end fixture per variant (Phase B exit criterion)

RFC #420 Phase B exit: "FL2VA and Ref2VA each have at least one sealed end-to-end fixture". This also needs:

- the encoder rows (`qwen3vl_layer50_contract`, `visual_vae_contract`, `audio_vae_contract`; the last two are done, on their own branches);
- `phase_b_full_chain`.

Proposed minimum, once those land:

1. **FL2VA, first+last keyframes, 768x1344, 124 frames, seed 42.**
   - Conditions from Diffusers `f53d552` (text encoder + VAE).
   - Layout and initial rows from `fl2va.build_fl2va_layout` / `fl2va_initial_rows`.
   - Run one strict model call with the `transformer/` weights.
   - Seal `packed_layout.layout_artifact(...)`, and run the trajectory artifact for 2 steps.
2. **Ref2VA, image + video-with-sound + audio, same size.**
   - Weights: `transformer_ref/`.
   - `map_transformer_ref(..., source_variant=identify_transformer_dir(...))`.
   - `check_layout_weights` before the call.
3. **Repeat each run (H-C0 / H-C7 at model level).**
   - Artifacts and velocities must be byte-equal.
   - Swapping the transformer partition must be refused before any forward (H16).

## 3. ROCm

There is no ROCm hardware on our side. The layout is built on the CPU and only moved to the device. The CUDA transfer test (`test_initial_rows_on_cuda_equal_cpu_bytes`) should be run once on a ROCm box: it uses `torch.cuda`, so it runs there unchanged.

## Send back

- The output of item 1.
- For item 2, once possible: the two artifacts (JSON), the per-step trajectory artifacts, and `pytest tests/models/minimax_h3/test_h3_{packed_layout,fl2va_pipeline,ref2va_pipeline}.py -rs` on the H100 node.

Push the results, with host names and paths stripped, as fork branch `handoff/h3-h100-pipeline-contracts-results-20261118` (folder `h100-results/pipeline-contracts-20261118/`). No PR and no comment.
