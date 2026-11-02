# 2. #588 on H100: resume check and preflight only (A and B)

**Scope for the 2 x 4 H100 setup: run A and B. C (candidate gate + three-seed generation) is not planned:**
it needs an H100 reference and rule that are not in the repository, and about 60 GPU hours. Keep C
below only as the record of why.

#588 body: the one-step preflight (128x128) ran on 4 x H100; **the three-seed generation (about 20 h
per seed on B200) was not run on H100.** Generation quality is "not passed" on B200 (seed 44 passes,
42 and 43 fail) against bounds that stay frozen. An H100 run would show whether the same seeds fail
on the other architecture. Do not touch `generation-contract.json`, the bounds or the thresholds.

## Why this is not just "submit the three sbatch files"
1. `generation.sbatch` refuses to start unless a `candidate.sbatch` job passed (`full-model-acceptance.json`
   status `passed`) for model paths byte-identical to the checkout. `candidate.sbatch` compares against
   `<reference_root>/fp32-reference` and the committed `full-model-ensemble-rule.json`, both of which are the
   **B200** reference and rule (k = 2.6).
2. #587 reports that on H100 the references and bounds were re-measured separately and the ensemble rule
   came out at k = 4.0 (evidence under `pr539/h100-full-model/` on the evidence branch at commit `23cc7e28`).
   That H100 reference and rule are not files in the repository, so the committed gate may fail on H100
   for rule reasons, not numerical ones. I have not run it.
3. All three `.sbatch` files request `--gres=gpu:b200:1` and 512 GB host memory. The 33B model is about
   66 GB in BF16; whether generation at 1344x768, 124 frames fits an H100 (93 GiB) is untested here.
   The H100 preflight used 4 GPUs. Override at submission, do not edit the files (clean tree is required):
   `sbatch --gres=gpu:<h100-gres>:1 ...`.

**Decision for Wenbo (before any 60 GPU-hour job):** (a) run only what is rule-independent (A and B below)
and report; (b) also run the gate on H100 under the committed rule and see whether it passes (C), then
generation; (c) skip. Recommendation: A + B first, they are cheap and answer the fit question.

## Checkout
Branch `feat/h3-generation-gate`, head `67365734`, clean tree. Strict build:
`RL_KERNEL_H3_STRICT=1 python setup.py build_ext --inplace` (ninja on PATH). Environment: torch with a
matching torchvision (the B200 reference used 2.9.1+cu130 / 0.24.1+cu130), `CUDA_HOME=/usr/local/cuda-13.0`,
`KERNEL_ALIGN_USE_FAST_MATH=0` (the scripts set it), provider source (the diffusers fork the runs pin),
an overlay with scikit-image and scipy for the comparison.

## A. Resume check on H100 (4 h, rule independent)
```bash
sbatch --gres=gpu:<h100-gres>:1 <site options> examples/models/minimax_h3/generation-resume-check.sbatch \
  <REPO> <python> <CKPT> <reference_image> <provider_source> <extra_deps> <result_root>
```
Expect the 8 files byte-equal between the uninterrupted and the resumed runs (as on B200).

## B. Preflight with memory/time readout (rule independent, minutes)
```bash
python -m tools.validation.models.h3_full_generation --checkpoint <CKPT> --reference-image <reference_image> \
  --candidate --preflight --output <OUT>/preflight
```
Send back the log, peak GPU memory (`torch.cuda.max_memory_allocated`, or `nvidia-smi` sampling) and the
seconds per sampling step. Then extrapolate: B200 is about 1500 s per step at full size. If a step is
more than about 1.5x that, or memory is within a few GiB of 93, stop here.

## C. Gate, then generation (only if the decision says so)
```bash
sbatch --gres=gpu:<h100-gres>:1 <site options> examples/models/minimax_h3/candidate.sbatch \
  <REPO> <python> <CKPT> <reference_root> <result_root> <reference_image> <provider_source> <extra_deps>
# if full-model-acceptance.json says "passed":
sbatch --gres=gpu:<h100-gres>:1 --array=0-2 <site options> examples/models/minimax_h3/generation.sbatch \
  <REPO> <python> <CKPT> <gate_job_id> <result_root> <reference_image> <provider_source> <extra_deps>
sbatch --dependency=afterok:<array_id> <site options> examples/models/minimax_h3/generation-compare.sbatch \
  <REPO> <python> <reference_root> <result_root> <extra_deps>
```
The 24 h limit is per attempt;
`--requeue` plus `--resume` continues from saved steps. Report per-seed PSNR / SSIM / temporal RMSE / audio
relative L2 and pass/fail against the unchanged bounds. If the gate in C fails on H100, send
`full-model-acceptance.json` and `full-model-comparison.json`; do not retune any bound.
