# H100 handoff T7: schedule, velocity step and the final Phase B chain (2026-11-24)

## Why

Three pieces of the RFC #738 H3-Image stack have not run on H100:
- `image_sigma_pdd_schedule` (#777);
- `image_velocity_step` (#778), including its Triton kernel;
- the last commit of `phase_b_image_full_chain` (#779), which moves the chain onto those two rows. T6 ran the commit before it.

T1–T6 are done and unchanged. Do not rerun them.

Placeholders are the same as in `../h3-image-20261123/README.md`: `<REPO>`, `<THIS_DIR>` (this folder), `<OUT>`, `<E1>`, `<V>` (with `accelerate==1.16.0`, from T6), `<IMG>` (with the T6 downloads) and `<CKPT>`. The rules there apply too: detached clean worktrees, change no file, idle GPU, report failures as they are, redact, `git commit -s`, no AI attribution.

Hardware: one idle H100 for all of T7 (the chain peaked at 67.2 GiB on B200).

## Code

Everything is on one public branch, so there is no bundle:

```bash
cd <REPO>
git fetch https://github.com/fusheng-ji/RL-Kernel.git feat/h3-image-phase-b
git rev-parse FETCH_HEAD        # must be a5989aede...
git worktree add --detach <W7> a5989aede
cd <W7>
git rev-parse HEAD^{tree}       # must equal the tree in reference/tree.txt
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E1>/bin/python setup.py build_ext --inplace > <OUT>/T7/build.log 2>&1
PYTHONPATH=<W7> <V>/bin/python -c "import os, rl_engine._C as c, triton; print(os.path.realpath(c.__file__).startswith(os.path.realpath('<W7>')), hasattr(c, 'conditioning_noise_mix_forward'), triton.__version__)" > <OUT>/T7/ext.txt
```

## T7

```bash
export RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT>
# a. the two new rows
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W7> <V>/bin/python -m pytest -q -p no:cacheprovider -rs \
  tests/models/minimax_h3_image/test_h3_image_pdd_schedule.py \
  tests/models/minimax_h3_image/test_h3_image_velocity_step.py \
  tests/ops/flow/test_h3_image_velocity_step.py > <OUT>/T7/pytest-new-rows.txt 2>&1
# b. the whole image suite at the stack head
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W7> <V>/bin/python -m pytest -q -p no:cacheprovider -rs \
  tests/models/minimax_h3_image tests/ops/flow/test_h3_image_velocity_step.py > <OUT>/T7/pytest-image.txt 2>&1
# c. the final chain on the released weights
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W7> <V>/bin/python tools/validation/models/h3_image_full_chain.py \
  --root <IMG> --text-encoder-root <CKPT> --out <OUT>/T7/run --repeat > <OUT>/T7/evidence.log 2>&1; echo "exit=$?" >> <OUT>/T7/evidence.log
# d. schedule and velocity-step numbers
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<W7> <V>/bin/python <THIS_DIR>/chart_data.py --out <OUT>/T7/data-h100.json > <OUT>/T7/data.log 2>&1; echo "exit=$?" >> <OUT>/T7/data.log
```

**Expected** (B200 values are in `reference/`):
- **ext.txt.** `True True` and a Triton version.
- **a.** B200 gave `66 passed`, 0 skipped. Nothing fails and nothing skips. A skip for missing Triton or CUDA means the row was not tested; send it as it is.
- **b.** B200 gave `460 passed`, 0 skipped. This includes the sealed H100 layer-50 text digests from round 1, so a mismatch in `test_release_fixtures_bitwise_against_diffusers` is a finding.
- **c.** `exit=0` and `all_gates_pass True`. Every fixture has `gate`, `secondary_equal` and `run_to_run_equal` true and `first_drift` null. In the `denoise` stage of each `*.rl-kernel.json`, the sampler `backend_id` is `H3ImageVelocityStepTritonOp`.
- **c, against T6.** The final latents and outputs should equal round 1's T6 on the same environment, as they did on B200. The comparison is made here.
- **d.** `exit=0`. In `velocity.steps`, `triton_vs_reference_differing` is 0 at every step. The sigmas and timesteps in `pdd_schedule` equal `reference/data-b200.json` exactly. Timings are reported, not compared.

## Sending results back

Add `results/T7/` on top of `handoff/h100-h3-image-results-20261123` (now `3dbfe6562`):
- `ext.txt`, `build.log`, `pytest-new-rows.txt`, `pytest-image.txt`, `evidence.log`, `data-h100.json`, `data.log`;
- `run/summary.json` and the eight `run/*.json` artifacts, without the `*.png` files (about 110 MB; their bytes are hashed in the artifacts);
- the `nvidia-smi` and environment records;
- one T7 line in `SUMMARY.md`.

Redact as before and push only that branch. No PR, no comment.
