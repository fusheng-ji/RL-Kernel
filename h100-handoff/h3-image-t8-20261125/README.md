# H100 handoff T8: PDD head contract and one real step (2026-11-25)

## Why

Two new rows of RFC #738 have run on B200 only:
- `h3_image_pdd_head_contract` (`984ee3d2e`);
- `ws1_one_h3_image_step` (`b685c4991`).

On B200 the FP32 boundary rows (`video_input_projection_fp32`, `video_output_projection_fp32`) and the three block GEMM rows (O, gate/up, down) are interim, because those rows are qualified on SM90 only. On H100 they resolve to the rows themselves, so this is the first run of the step on them.

T1-T7 are separate. Do not rerun them. Placeholders and rules are those of `../h3-image-20261123/README.md`.

Hardware: one idle H100 (94 GB). The step peaked at 79.4 GiB on B200 (two-reference edit, checkpointed training step). Do not pass `--plain-autograd`: it needs 111 GiB.

## Code

The commits are not public yet; they are in the bundle, on top of the public `feat/h3-image-phase-b` head `a5989aede`.

```bash
cd <REPO>
git fetch https://github.com/fusheng-ji/RL-Kernel.git feat/h3-image-phase-b
git bundle verify <THIS_DIR>/h3-image-t8-20261125.bundle
git fetch <THIS_DIR>/h3-image-t8-20261125.bundle feat/h3-image-one-step
git rev-parse FETCH_HEAD        # must be b685c4991...
git worktree add --detach <W8> FETCH_HEAD
cd <W8>
git rev-parse HEAD^{tree}       # must equal reference/tree.txt
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E1>/bin/python setup.py build_ext --inplace > <OUT>/T8/build.log 2>&1
PYTHONPATH=<W8> <V>/bin/python -c "import os, rl_engine._C as c, rl_engine._H3_C as h, triton; p=os.path.realpath; print(p(c.__file__).startswith(p('<W8>')), h.h3_attention_build_info()['profile'], triton.__version__)" > <OUT>/T8/ext.txt
```

## T8

```bash
export RL_KERNEL_H3_IMAGE_ROOT=<IMG> RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYT="env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W8> <V>/bin/python"
TOOL=tools/validation/models/h3_image_step.py
# a. the new tests
$PYT -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3_image/test_h3_image_pdd_head.py \
  tests/models/minimax_h3_image/test_h3_image_step.py > <OUT>/T8/pytest-new.txt 2>&1
# b. the whole image suite
$PYT -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3_image > <OUT>/T8/pytest-image.txt 2>&1
# c. capture the real inputs, then replay
$PYT $TOOL capture --root <IMG> --text-encoder-root <CKPT> --out <OUT>/T8/inputs > <OUT>/T8/capture.log 2>&1; echo "exit=$?" >> <OUT>/T8/capture.log
$PYT $TOOL replay --root <IMG> --inputs <OUT>/T8/inputs --out <OUT>/T8/run-t2i --fixtures t2i_upscaled > <OUT>/T8/run-t2i.log 2>&1; echo "exit=$?" >> <OUT>/T8/run-t2i.log
$PYT $TOOL replay --root <IMG> --inputs <OUT>/T8/inputs --out <OUT>/T8/run-edit --fixtures edit_two_refs > <OUT>/T8/run-edit.log 2>&1; echo "exit=$?" >> <OUT>/T8/run-edit.log
# d. a second process reproduces the first
$PYT $TOOL replay --root <IMG> --inputs <OUT>/T8/inputs --out <OUT>/T8/run-b --fixtures t2i_upscaled --steps 0,5 --against <OUT>/T8/run-t2i > <OUT>/T8/run-b.log 2>&1; echo "exit=$?" >> <OUT>/T8/run-b.log
```

**Expected** (B200 values are in `reference/`):
- **ext.txt.** `True h3-cuda-strict-v1` and a Triton version.
- **a.** B200 gave `83 passed`, 0 skipped. `test_cuda_binding_is_the_row_on_sm90_and_interim_elsewhere` now takes its SM90 branch.
- **b.** B200 gave `514 passed`, 0 skipped. This includes the sealed H100 layer-50 text digests, so a mismatch there is a finding.
- **c.** `exit=0` and `ok True` for both fixtures. In `summary.json`, `proj_in`, `pdd_proj_out`, `block.o_proj`, `block.ffn_gate_up` and `block.ffn_down` have status `row`. Every entry of `controls` is null except `next_head_first_drift`, whose node is `pdd_proj_out`, and `training.ok` is true.
- **c, capture.** The layouts and timesteps are the same as on B200. The image rows of steps 1-5 may differ from B200: they come from the Diffusers loop on this GPU.
- **d.** `exit=0`, `equal_to_other_process` true for both steps.
- Distances to Diffusers and timings are reported, not compared.

A failure in the GEMM rows under checkpointing or in `canonical_backward_session` is a finding: send it as it is.

## Sending results back

Add `results/T8/` on top of `handoff/h100-h3-image-results-20261123`: `ext.txt`, `build.log`, both pytest files, the logs, every `summary.json` and `*.digests.json`, `inputs/*.json` (not the `*.safetensors`), the `nvidia-smi` and environment records, and one T8 line in `SUMMARY.md`. Redact as before and push only that branch. No PR, no comment.
