# H3 full-model candidate: acceptance evidence (B200)

All runs use one B200, torch 2.9.1+cu130 and diffusers `f53d552`, from clean checkouts.
The reference was regenerated on 2026-10-16 after the first run's artifacts were deleted
(see `examples/models/minimax_h3/README.md`); the regenerated run reproduces 1,614 of the
1,635 pinned files byte for byte and the same bounds and rule.

## Numerical acceptance (passed)

Gates of `candidate.sbatch` at `c249a735` against the regenerated reference.

- `full-model-acceptance.json` -- the decision: the provider-ensemble rule
  (`examples/models/minimax_h3/full-model-ensemble-rule.json`, relative_l2 only) applied to the
  candidate. `status: passed`, 0 of 764 tensors failing, worst tensor at 0.55 of its bound.
- `full-model-comparison.json` -- the candidate against the frozen FP32 reference, per tensor
  (`max_abs`, `relative_l2`). Its `status: failed` is the *original single-configuration* bounds
  (`full-model-thresholds.json`, 1.5 x one provider pass + floor), kept as a record: 54 of 764
  tensors fail them, 52 on `max_abs` only and 2 on `relative_l2` as well.
- `first-reference/` -- the same two files from the first reference run (`4792c8a9`). The numbers are
  identical; only the pinned hashes differ.
- `provider-spread-summary.json` -- why the single-configuration rule was replaced: those bounds
  applied to six diffusers provider runs (four distinct configurations) compared with the same
  reference. Providers that differ only in SDPA backend or BLAS library fail 36-66 of them on
  `max_abs`, the default configuration they were calibrated on none; the candidate's 54 are in that
  range. Made with `tools/validation/models/h3_full_provider_spread.py` at `c8ac5b19`.

The ensemble rule is calibrated on provider configurations only. It was designed after the
candidate had failed the single-configuration bounds, which is why that comparison is kept.

## Generation quality (not passed)

- `generation-quality-comparison.json` -- the candidate's three seeds (124 frames, 1344x768)
  against the reference generation with the fixed bounds (PSNR >= 30 dB, SSIM >= 0.99, temporal RMSE
  <= 0.01, audio relative L2 <= 0.05): seed 44 passes, seeds 42 and 43 do not. The bounds were fixed
  before candidate generation and are unchanged.
- `provider-variant-generation-quality.json` -- the same scoring for the provider itself with one SDPA
  backend forced on the transformer (flash, memory-efficient), same seeds and inputs: 0 of 3 seeds
  pass for either; audio relative L2 exceeds 0.05 on every seed (0.054-0.264). The bounds had only
  been checked against a bitwise-identical repeat of the provider (PSNR 300 dB).
- `first-step-replay.json` -- the first denoising step replayed from its saved packed inputs under
  different configurations and compared with the provider's saved step. Replaying the provider is
  bitwise equal to it. The candidate's video relative L2 is 0.021-0.029; the provider with flash or
  memory-efficient attention gives 0.022; swapping any single group of block nodes to their
  provider replay leaves 0.019-0.023 and only swapping all of them gives 0.

The candidate is not uniformly better than the provider variants: on seeds 42 and 44 its PSNR and
temporal RMSE are slightly worse than both variants', on seed 43 better.
