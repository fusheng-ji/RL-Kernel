# H100 handoff T10: T8 / T9 reruns after the two SM90 fixes (2026-11-26)

## Why

T8 and T9 found two things that only show on H100, where the GEMM and projection rows bind (on B200 they are interim):

1. **`h3_ffn_down_gemm` row limit.** The row declares at most 32,768 rows. The batch-position control batches two edit requests (2 x 17,203 = 34,406 rows), so T8 `run-edit` stopped with `ValueError: row count must be in [1,32768]`. T9-edit runs the same control and would have stopped there too.
2. **Strided input to the PDD head projection.** `test_cuda_projection_is_accurate_and_row_invariant` calls a bound head on a slice (`x[:, 100:101]`). The SM90 `video_output_projection_fp32` row raises `x must be contiguous`; the B200 interim GEMM accepted it.

Fixes, both new signed-off commits (nothing was rewritten; `69e0e42c9` is an ancestor):
- `d3bc9fe41`: the bound head passes `hidden.contiguous()` to the projection (a no-op for contiguous rows).
- `f4171b776`: the strict transformer wraps the down row in `RowSlicedGemm`. Without a weight gradient it runs consecutive slices of at most 32,768 rows with the matching logical keys. The row's forward has fixed 64x64x32 tiles, no split-K and a fixed K order, so each output row depends on its own input row only. With `weight.requires_grad` the row is called once and keeps its own limit.

Head: `feat/h3-image-ws1-full` at `4cfa3cbfc` (merges of the two fixes into the stack).

This T10 replaces the failed parts of T8 and T9 only. T7, T8 `run-t2i` and T9-t2i stay as they are. Placeholders and rules are those of `../h3-image-20261123/README.md`: `<REPO>`, `<THIS_DIR>`, `<OUT>`, `<E1>`, `<V>`, `<IMG>`, `<CKPT>`. Detached clean worktree, change no file, idle GPU, report failures as they are, redact, `git commit -s`, no AI attribution.

## Hardware and time

Two single-GPU jobs. Each needs an idle **94 GB** H100 and at least 128 GB of host memory. On a shared node use the `GPU=${CUDA_VISIBLE_DEVICES}` block of T9 and say in `SUMMARY.md` which card it was.

| Job | What | B200 time |
|---|---|---|
| T10-a | build, image suite, T8 capture + edit replay, T8 second process, T9 t2i bytes check | about 1 h 30 |
| T10-edit | T9-edit | 2 h 10, peak 81.6 GiB |

**Give T10-edit a time limit of at least 6 hours.** The T9-edit attempt was terminated before its first fixture finished; H100 is slower than B200 on this workload (T9-t2i `t2i_example` strict chain: 579 s vs 481 s).

## Code

```bash
cd <REPO>
git fetch https://github.com/fusheng-ji/RL-Kernel.git feat/h3-image-phase-b
git bundle verify <THIS_DIR>/h3-image-t10-20261126.bundle
git fetch <THIS_DIR>/h3-image-t10-20261126.bundle feat/h3-image-ws1-full
git rev-parse FETCH_HEAD        # must be 4cfa3cbfca73772c1f3f27af0aca50a4e6ab10cc
git worktree add --detach <W10> FETCH_HEAD
cd <W10>
git rev-parse HEAD^{tree}       # must equal reference/tree.txt
mkdir -p <OUT>/T10
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E1>/bin/python setup.py build_ext --inplace > <OUT>/T10/build.log 2>&1
PYTHONPATH=<W10> <V>/bin/python -c "import os, rl_engine._C as c, rl_engine._H3_C as h, triton; p=os.path.realpath; print(p(c.__file__).startswith(p('<W10>')), h.h3_attention_build_info()['profile'], triton.__version__)" > <OUT>/T10/ext.txt
```

Build once; both jobs use `<W10>`. **Run every command below from `<W10>`** (the T8 `run-b` failure was a run from another directory).

## T10-a

```bash
cd <W10>
export RL_KERNEL_H3_IMAGE_ROOT=<IMG> RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYT="env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=<W10> <V>/bin/python"
O=<OUT>/T10/a; mkdir -p $O
# a. the whole image suite (includes the projection test and the new row-slicing tests)
$PYT -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3_image > $O/pytest-suite.txt 2>&1
# b. T8 again: capture, the edit replay that stopped, and the second process
STEP=tools/validation/models/h3_image_step.py
$PYT $STEP capture --root <IMG> --text-encoder-root <CKPT> --out $O/inputs > $O/capture.log 2>&1; echo "exit=$?" >> $O/capture.log
$PYT $STEP replay --root <IMG> --inputs $O/inputs --out $O/run-edit --fixtures edit_two_refs > $O/run-edit.log 2>&1; echo "exit=$?" >> $O/run-edit.log
$PYT $STEP replay --root <IMG> --inputs $O/inputs --out $O/run-t2i --fixtures t2i_upscaled > $O/run-t2i.log 2>&1; echo "exit=$?" >> $O/run-t2i.log
$PYT $STEP replay --root <IMG> --inputs $O/inputs --out $O/run-b --fixtures t2i_upscaled --steps 0,5 --against $O/run-t2i > $O/run-b.log 2>&1; echo "exit=$?" >> $O/run-b.log
# c. the fixes do not change the T9 t2i bytes on H100: strict t2i_upscaled against T9's run-a
TRAJ=tools/validation/models/h3_image_trajectory.py
$PYT $TRAJ --root <IMG> --text-encoder-root <CKPT> --out $O/t2i-vs-t9 --fixtures t2i_upscaled --strict-only --against <OUT>/T9/t2i/run-a > $O/t2i-vs-t9.log 2>&1; echo "exit=$?" >> $O/t2i-vs-t9.log
```

## T10-edit

The T9-edit commands, at `4cfa3cbfc`:

```bash
cd <W10>
export RL_KERNEL_H3_IMAGE_ROOT=<IMG> RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYT="env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=<W10> <V>/bin/python"
TRAJ=tools/validation/models/h3_image_trajectory.py
O=<OUT>/T10/edit; mkdir -p $O
$PYT $TRAJ --root <IMG> --text-encoder-root <CKPT> --out $O/run-a \
  --fixtures edit_two_refs,edit_example --controls edit_two_refs --train-steps 5 > $O/run-a.log 2>&1; echo "exit=$?" >> $O/run-a.log
$PYT $TRAJ --root <IMG> --text-encoder-root <CKPT> --out $O/run-b \
  --fixtures edit_two_refs,edit_example --strict-only --against $O/run-a > $O/run-b.log 2>&1; echo "exit=$?" >> $O/run-b.log
```

## Expected

- **ext.txt.** `True h3-cuda-strict-v1` and a Triton version.
- **a, pytest-suite.** B200 at `4cfa3cbfc`: `535 passed`, 0 skipped (`reference/b200-suite.txt`; T9 had 531, plus three row-slicing tests and one head test). No failure; in particular `test_cuda_projection_is_accurate_and_row_invariant` passes.
- **b.** `exit=0` and `ok True` for `run-edit` and `run-t2i`; `binding_counts` row 30 / interim 6 / reference 3 as in T8 `run-t2i`. In `run-edit/summary.json` every `controls` entry is null except `next_head_first_drift` (node `pdd_proj_out`), and `training.ok` is true. `run-b`: `exit=0`, `equal_to_other_process` true for steps 0 and 5.
- **c.** `exit=0`, `ok True`, `equal_to_other_process` true: the head and down-row changes leave every step record and the output bytes of T9 unchanged (`t2i_upscaled` stays within 32,768 rows even batched). On B200 the same check against `69e0e42c9` gave `equal_to_other_process` true (`reference/b200-t2i-vs-69e0e42c9.summary.json`).
- **T10-edit.** As T9's "Expected" for the edit job: `ok` true, bindings 30 / 6 / 3, `controls.ok` true with every drift null, the training step at 5 ok, `fixed_reference_rows.fixed` true at all six steps (6,528 and 1,024 rows), `schedule_equal` true; `run-b` `equal_to_other_process` true for both fixtures. B200 values: `../h3-image-t9-20261126/reference/b200-edit/`. Distances, PSNR, peaks and timings are reported, not compared.

A failure in the row slices, in the checkpointed backward or in the second process is a finding: send it as it is.

## Sending results back

Add `results/T10/` on top of `handoff/h100-h3-image-results-20261123`; fetch its tip first and never force-push. Include `ext.txt`, `build.log`, every log and pytest file, `nvidia-smi.txt` and the environment record per job, every `summary.json`, `*.digests.json`, `*.records.json` and the other `*.json` under the run folders, and `inputs/*.json`. Not the `*.png` or `*.safetensors` files. One T10 line in `SUMMARY.md`. Redact as before (no user name, project id, node or absolute path in any file, logs included) and push only that branch. No PR, no comment.
