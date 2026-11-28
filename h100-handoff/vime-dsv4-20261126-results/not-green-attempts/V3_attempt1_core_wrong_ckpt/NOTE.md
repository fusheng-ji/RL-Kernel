# V3 (SUITE=core), 2026-11-28 01:50–01:54Z, job <JOB_1> — not green, scheduling error

All four cases (T0-50, T1-50, T2-50, skip-T1-2) exited rc=1 within ~40 s, before step 0:
`OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '<SCRATCH>/h3ckpt/hf_full'`.

Cause: the schedule's `scripts/common.sh` exports `CKPT=$SCRATCH_ROOT/h3ckpt/hf_full` (H3 diffusers weights) and the
handoff's `common.sh` uses `CKPT=${CKPT:-$W/ckpt/v4-4l}`, so V3 inherited the wrong checkpoint. Not a vime result.
V4 (drift.sbatch sets CKPT per case) was not affected. Replays were skipped (no runs/core/roll/0.pt).

Fix: `scripts/u_vime.sh` pins `CKPT=$W/ckpt/v4-4l` for V3/V4. Rerun as SUITE=core-r2 (runs/core-r2, out/vime/V3-core-r2*).
The original run directory vime/runs/core is kept unchanged.
