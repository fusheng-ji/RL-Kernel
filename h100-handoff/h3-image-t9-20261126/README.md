# H100 handoff T9: six strict steps end to end, T2I and edit (2026-11-26)

## Why

Two rows of RFC #738 have run on B200 only:
- `ws1_full_h3_image_t2i`;
- `ws1_full_h3_image_edit`.

Both are in `69e0e42c9`, one commit on top of T8's `b685c4991`. The six-step strict loop runs on the released weights, from the prompt to the output bytes, and it records every step. The controls check batching, permutation, a second request in the batch, the loop run alone, a checkpointed training step on the recorded rollout, and a second process.

On B200, the FP32 boundary rows (`proj_in`, `pdd_proj_out`) and the three GEMM rows (O, gate/up, down, in the blocks and in the refiner) are interim, because those rows are qualified on SM90 only. On H100 they resolve to the rows themselves. This is the first run of the whole trajectory on them.

T1–T8 are separate. Do not rerun them. Placeholders and rules are those of `../h3-image-20261123/README.md`: `<REPO>`, `<THIS_DIR>` (this folder), `<OUT>`, `<E1>`, `<V>` (with `accelerate==1.16.0`, from T6), `<IMG>` (with the T6 downloads) and `<CKPT>`. The rules apply too: detached clean worktree, change no file, idle GPU, report failures as they are, redact, `git commit -s`, no AI attribution.

## Hardware and time

T9 is two single-GPU jobs, **T9-t2i** and **T9-edit**. They can run one after the other on one card, or at the same time on two cards. Each card must be an idle **94 GB** H100.

| Job | Fixtures | Peak on B200 | Time on B200 |
|---|---|---|---|
| T9-t2i | `t2i_upscaled` (controls, training at all six steps), `t2i_example`, then the tests | 69.6 GiB (`t2i_example`) | 1 h 02 min, including a 14 min build |
| T9-edit | `edit_two_refs` (controls, training at step 5), `edit_example` | **81.6 GiB** (training step on `edit_two_refs`) | 2 h 10 min |

T9-edit's peak leaves about 12 GiB on a 94 GB card; on an 80 GB card it does not fit. Expect H100 to be slower than B200. Ask for at least 128 GB of host memory per job. Out of memory is a finding: send it as it is, and change no code.

**Shared nodes.** On a node with other users' jobs, set `GPU` to the allocated card, not to a fixed `0`:

```bash
GPU=${CUDA_VISIBLE_DEVICES:?no GPU allocated}
case "$GPU" in *,*) echo "more than one GPU allocated: $GPU"; exit 1;; esac
mkdir -p <OUT>/T9/<job>      # <job> is t2i or edit
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv -i "$GPU" > <OUT>/T9/<job>/nvidia-smi.txt   # memory.used must be 0 MiB
```

On a card you have to yourself, `GPU=0`. Say in `SUMMARY.md` which one it was, and how many other GPUs on the node were busy (the count only).

## Code

The commit is not public yet. It is in the bundle, on top of the public `feat/h3-image-phase-b` head `a5989aede` (the same base as T8).

```bash
cd <REPO>
git fetch https://github.com/fusheng-ji/RL-Kernel.git feat/h3-image-phase-b
git bundle verify <THIS_DIR>/h3-image-t9-20261126.bundle
git fetch <THIS_DIR>/h3-image-t9-20261126.bundle feat/h3-image-ws1-full
git rev-parse FETCH_HEAD        # must be 69e0e42c96eba0b3aac2202c833520a18f27277a
git worktree add --detach <W9> FETCH_HEAD
cd <W9>
git rev-parse HEAD^{tree}       # must equal reference/tree.txt
mkdir -p <OUT>/T9
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E1>/bin/python setup.py build_ext --inplace > <OUT>/T9/build.log 2>&1
PYTHONPATH=<W9> <V>/bin/python -c "import os, rl_engine._C as c, rl_engine._H3_C as h, triton; p=os.path.realpath; print(p(c.__file__).startswith(p('<W9>')), h.h3_attention_build_info()['profile'], triton.__version__)" > <OUT>/T9/ext.txt
```

Build once. Both jobs use the same `<W9>`. The two jobs write to different folders, so they can share it.

## T9-t2i

```bash
cd <W9>
export RL_KERNEL_H3_IMAGE_ROOT=<IMG> RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYT="env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=<W9> <V>/bin/python"
TOOL=tools/validation/models/h3_image_trajectory.py
O=<OUT>/T9/t2i; mkdir -p $O
# a. the strict trajectory, the Diffusers chain and its variants, the controls and the training steps
$PYT $TOOL --root <IMG> --text-encoder-root <CKPT> --out $O/run-a \
  --fixtures t2i_upscaled,t2i_example --controls t2i_upscaled --train-steps 0,1,2,3,4,5 > $O/run-a.log 2>&1; echo "exit=$?" >> $O/run-a.log
# b. a second process reproduces every strict step record and the output bytes
$PYT $TOOL --root <IMG> --text-encoder-root <CKPT> --out $O/run-b \
  --fixtures t2i_upscaled,t2i_example --strict-only --against $O/run-a > $O/run-b.log 2>&1; echo "exit=$?" >> $O/run-b.log
# c. the new tests, then the whole image suite
$PYT -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3_image/test_h3_image_ws1_full.py > $O/pytest-new.txt 2>&1
$PYT -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3_image > $O/pytest-suite.txt 2>&1
```

## T9-edit

```bash
cd <W9>
export RL_KERNEL_H3_IMAGE_ROOT=<IMG> RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYT="env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=<W9> <V>/bin/python"
TOOL=tools/validation/models/h3_image_trajectory.py
O=<OUT>/T9/edit; mkdir -p $O
$PYT $TOOL --root <IMG> --text-encoder-root <CKPT> --out $O/run-a \
  --fixtures edit_two_refs,edit_example --controls edit_two_refs --train-steps 5 > $O/run-a.log 2>&1; echo "exit=$?" >> $O/run-a.log
$PYT $TOOL --root <IMG> --text-encoder-root <CKPT> --out $O/run-b \
  --fixtures edit_two_refs,edit_example --strict-only --against $O/run-a > $O/run-b.log 2>&1; echo "exit=$?" >> $O/run-b.log
```

## Expected

The B200 values are in `reference/b200-t2i/` and `reference/b200-edit/` (the `summary.json` of each run).

- **ext.txt.** `True h3-cuda-strict-v1` and a Triton version.
- **run-a.** `exit=0`, and the log ends in `ok True`. Each fixture prints `"six_in_order": true`. In `summary.json`:
  - `ok` and `heads_released` are true. `sampler.backend_id` is `H3ImageVelocityStepTritonOp`.
  - **Bindings, the new part.** `proj_in`, `pdd_proj_out`, `block.o_proj`, `block.ffn_gate_up`, `block.ffn_down`, `refiner.o_proj`, `refiner.ffn_gate_up` and `refiner.ffn_down` have status `row` (on B200, `interim`). `binding_counts` should then be `row 30`, `interim 6` (`q_norm`, `k_norm` and `swiglu` in the blocks and in the refiner), `reference 3` (B200: 22 / 14 / 3).
  - **Controls** (on `t2i_upscaled` / `edit_two_refs`): `controls.ok` is true. `permuted_first_drift`, `loop_alone.first_drift`, and `request_first_drift` / `partner_first_drift` of both `batch_row_*` are null. `partner.differs_from_request` is true.
  - **Training.** For every step in `--train-steps`: `ok` and `other_heads_have_no_gradient` are true, `gradients_equal.checkpointed_again` is true, `forward_first_drift_vs_rollout` is null for both checkpointed passes, and every entry of `finite_and_nonzero` is true.
  - **Edit only.** `strict.fixed_reference_rows.fixed` is true and `steps_equal_to_conditioned` is true at all six steps: the reference rows (6,528 for `edit_two_refs`, 1,024 for `edit_example`) stay fixed. In the controls, `partner.fixed_reference_rows` is true too.
  - **Provider.** `schedule_equal` is true and `input_mismatches` is empty. The strict trajectory is **not** bitwise equal to Diffusers, on B200 either: the first drift is the step-0 velocity, and the chain is chaotic, so the distance grows over the six steps. `strict_vs_provider` and `against_variants` are reported, not compared. On B200, the strict output PSNR against Diffusers' default arm was at or above the lowest Diffusers variant for all four fixtures: 22.8 vs 22.5 dB (`t2i_upscaled`), 20.3 vs 20.1 (`t2i_example`), 30.3 vs 28.2 (`edit_two_refs`), 37.8 vs 36.5 (`edit_example`). The comparison with B200 is made here. On H100 a variant may report `unavailable` (a kernel this build or sequence length does not have) or `equal_to_default`; both are results.
- **run-b.** `exit=0`, `ok True`, and `equal_to_other_process` is true for every fixture.
- **pytest-new.** B200 gave `17 passed`, 0 skipped.
- **pytest-suite.** B200 gave `531 passed`, 0 skipped. It includes the sealed H100 layer-50 text digests, so a mismatch there is a finding.
- **Digests.** `final_latent_sha256` and `output_sha256` are expected to differ from B200, because BF16 matmuls are hardware-specific. Distances, PSNR, peaks and timings are reported, not compared.

A failure in a GEMM row, in the checkpointed backward, or in the second process is a finding: send it as it is.

## Sending results back

Add `results/T9/` on top of `handoff/h100-h3-image-results-20261123`. Fetch its tip first, and never force-push: T7 and T8 may have landed there in the meantime. Include:
- `ext.txt` and `build.log`;
- for each job (`t2i/`, `edit/`): the logs, the pytest files, `nvidia-smi.txt`, and every `*.json` under `run-a/` and `run-b/`. **Do not** include the `*.png` files (up to about 30 MB each; their bytes are hashed in the JSON);
- the environment record (torch / CUDA / driver / GPU name);
- one T9 line in `SUMMARY.md`.

Redact as before and push only that branch. No PR, no comment.
