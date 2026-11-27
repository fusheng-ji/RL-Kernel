# H100 handoff T11: the T10-edit second process (2026-11-27)

## Why

T10-edit ran `run-a` only; its `run-b` line did not run (results commit `9991f16fa`, "T10-edit run-b not run"). Without it the edit rows have no H100 evidence that a second process gives the same step records and output. T11 is that one command, nothing else.

Head: `feat/h3-image-ws1-full` at `4cfa3cbfc`, the same tree as T10 (`reference/tree.txt`). Placeholders and rules are those of `../h3-image-20261123/README.md` and `../h3-image-t10-20261126/README.md`: detached clean worktree, change no file, idle GPU, report failures as they are, redact, `git commit -s`, no AI attribution.

## Hardware and time

One single-GPU job: an idle **94 GB** H100 and at least 128 GB of host memory. Strict only, no controls, no training step: T10-edit's strict chains took 610 s and 731 s, so expect about 30-45 min with loading. **Time limit 2 hours.** On a shared node use the `GPU=${CUDA_VISIBLE_DEVICES}` block of T9 and say in `SUMMARY.md` which card it was.

## Code

If the T10 worktree `<W10>` is still there and `git status --porcelain` in it is empty, use it as it is (no rebuild). Otherwise recreate it with the "Code" block of `../h3-image-t10-20261126/README.md` (bundle, `4cfa3cbfc`, tree check, build, `ext.txt`) and write `build.log` / `ext.txt` to `<OUT>/T11/`.

The run compares against T10-edit's `run-a` step records. Use the copy on the results branch, so the comparison is against exactly what was published:

```bash
cd <REPO>
git fetch https://github.com/fusheng-ji/RL-Kernel.git handoff/h100-h3-image-results-20261123
git rev-parse FETCH_HEAD                 # 9991f16fa... or a later commit on top of it
mkdir -p <OUT>/T11/against
git archive FETCH_HEAD results/T10/edit/run-a | tar -x -C <OUT>/T11/against
ls <OUT>/T11/against/results/T10/edit/run-a/edit_two_refs.records.json <OUT>/T11/against/results/T10/edit/run-a/edit_example.records.json
```

## T11

```bash
cd <W10>
git status --porcelain | wc -l > <OUT>/T11/status-before.txt     # must be 0
export RL_KERNEL_H3_IMAGE_ROOT=<IMG> RL_KERNEL_H3_IMAGE_DIR=<IMG> RL_KERNEL_H3_IMAGE_VAE_DIR=<IMG>/vae_diffusers RL_KERNEL_H3_IMAGE_TEXT_ENCODER_ROOT=<CKPT>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYT="env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=<W10> <V>/bin/python"
TRAJ=tools/validation/models/h3_image_trajectory.py
O=<OUT>/T11; nvidia-smi > $O/nvidia-smi.txt
$PYT -m pip list 2>/dev/null | grep -E "^(torch|triton|diffusers|transformers|accelerate) " > $O/env.txt
$PYT $TRAJ --root <IMG> --text-encoder-root <CKPT> --out $O/run-b \
  --fixtures edit_two_refs,edit_example --strict-only --against $O/against/results/T10/edit/run-a > $O/run-b.log 2>&1; echo "exit=$?" >> $O/run-b.log
git status --porcelain | wc -l > $O/status-after.txt             # must be 0
```

## Expected

- `run-b.log`: `exit=0`, `ok True`, `six_in_order` true for both fixtures.
- `run-b/summary.json`: `binding_counts` row 30 / interim 6 / reference 3; `fixtures.edit_two_refs.equal_to_other_process` and `fixtures.edit_example.equal_to_other_process` both true; `strict.fixed_reference_rows.fixed` true for both. On B200 the same check gave true for both (`../h3-image-t9-20261126/reference/b200-edit/`).
- `status-before.txt` and `status-after.txt`: 0.

Timings and peaks are reported, not compared. `equal_to_other_process` false is a finding: send it as it is, with `run-b/*.records.json`.

## Sending results back

Add `results/T11/` on top of `handoff/h100-h3-image-results-20261123`; fetch its tip first and never force-push. Include `run-b.log`, `run-b/summary.json`, `run-b/*.records.json` and the other `*.json` under `run-b/`, `nvidia-smi.txt`, `env.txt`, both status files, and `build.log` / `ext.txt` if you rebuilt. Not the `against/` copy, not `*.png` or `*.safetensors`. One T11 line in `SUMMARY.md`.

Redact as before (no user name, project id, node or absolute path in any file, logs included). **The job-id rule must not touch data:** replace job ids only where they appear as a job id (for example after `SLURM_JOB_ID=` or in a job-name/log-name field), never a bare 7-digit number. After redacting, check that every `*.json` parses and that no `<JOBID>` is left in any JSON file. Push only that branch. No PR, no comment.
