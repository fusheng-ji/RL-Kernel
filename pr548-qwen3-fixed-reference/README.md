# Qwen3-8B fixed-rollout reference for PR #548 (frozen before any candidate run)

Kept on the `pr_assets` branch instead of in the PR so the PR stays small; the PR body links here, pinned to the commit.

These files are the "acceptance manifest and per-tensor error bounds" that
`examples/post_training/tp_ppo/QWEN3_ACCEPTANCE.md` requires to be committed before
`qualify_candidate.sbatch` is submitted. No candidate run has been made; nothing here
was derived from, or can be widened by, a candidate result.

| File | Content |
|---|---|
| `acceptance.json` | `status: frozen`; sha256 of all 12,866 artifacts of the reference run (rollout, targets, the eight HF control reports and their tensors, the bounds files); the behavior-logprob tolerance (atol 0.41578529907226564, rtol 0); the qualification matrix |
| `{actor,critic}-fp32-repeat-bounds.json` | bounds from two independent FP32 controls on the same rollout and targets |
| `{actor,critic}-bf16-repeat-bounds.json` | the same for BF16 |
| `{actor,critic}-cross-precision-bounds.json` | FP32 against BF16 |

Each bounds file holds the per-tensor error bounds (1,604 tensors for the actor, 1,608 for
the critic) under the rule `4 * independent control max_abs error + metric floor` (floors:
1e-5 scores, 1e-6 losses/gradients/norms, 1e-7 parameter updates).

## How they were produced
`qualify_reference.sbatch` at source commit `17d19247549ac5a480a83befc35abbce504ae390`
(clean checkout), four B200 GPUs (driver 580.126.20, CUDA 13.0): actual vLLM TP2 sampling
captured the rollout, HF scored it independently, and both HF precisions were repeated twice on
the same immutable rollout and targets; `freeze_fixed_reference` then wrote `acceptance.json`.
Model: `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`. Environment: HF scoring
with torch 2.9.1+cu130 and transformers 5.13.1; vLLM environment with vllm 0.15.1, torch 2.9.1
and transformers 4.57.6.

Anchors: rollout manifest `bcef65787419…`, targets manifest `098a6ef95134…`, FP32 reference
report `1f5680ac556d…` (first 12 hex digits; the full values are in `acceptance.json`).

## Which pass is the reference
The reference job was preempted and requeued three times. The first two passes did not finish.
The third wrote its acceptance manifest and was preempted at that moment, so the job reran
everything; the fourth pass ran to completion and is the job's final result, and its files are the
ones frozen here. The third pass had sampled a different vLLM rollout (its rollout, targets and
FP32 reference hashes differ) and produced its own bounds; those were **not** used. The two
passes share 12,850 of the 12,866 pinned files byte for byte; the 16 that differ are small
manifests, reports and bounds files.

## Verifying
The raw evidence (about 955 GiB of per-parameter tensors) stays outside Git and is anchored by
`acceptance.json`. With the raw reference directory available:

    python - <<'PY'
    from pathlib import Path
    from examples.post_training.tp_ppo.fixed_contract import verify_reference
    verify_reference(Path("<reference dir>"),
                     Path("<this directory>/acceptance.json"))
    PY

This was run against the raw data with exactly these committed files: all 12,866 artifacts
match, and each committed bounds file matches its pinned hash. Candidate runs pass this
`acceptance.json` to `qualify_candidate.sbatch` and read the bounds from the reference directory.
