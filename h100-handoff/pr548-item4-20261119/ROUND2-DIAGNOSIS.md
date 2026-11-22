# Round 2 diagnosis (2026-11-22): the one TP4 update over its bound

Round 2's step 6 stopped at run 2 (results `7dabfb7d`, `23001bb1`). The Megatron FP32 actor at
TP4 passed 1603 of 1604 tensors. It failed one:

| Tensor | TP4 | TP2 | bound |
|---|---:|---:|---:|
| `0/update/model.layers.17.post_attention_layernorm.weight` | 7.749e-07 (2.29x) | 2.384e-07 (0.70x) | 3.384e-07 |
| `0/gradient/model.layers.17.post_attention_layernorm.weight` | 3.196e-06 (0.30x) | 2.354e-06 (0.22x) | 1.060e-05 |

The gradient of the same tensor is well inside its bound. The error of the update sits in about one
element: its L2 norm (1.22e-3 of the update's norm) equals its max.

**Hypothesis: AdamW's first step amplifies a small gradient difference at an element whose
clipped gradient is near `eps`.**
- At the first step AdamW moves each weight by about `-lr * g_c / (|g_c| + eps)`, with `g_c` the
  clipped gradient. That is `±lr` wherever `|g_c|` is far above `eps`, so differences in `g` do
  not show there.
- Where `|g_c|` is near or below `eps` the update is linear in `g`. Its slope is up to
  `lr / (eps * grad_norm) = 1e-5 / (1e-8 * 76.92) ≈ 13` in terms of the unclipped gradient.
- So a gradient difference of only 6e-8 at such an element moves its update by up to 13 units of
  2^-24 = 7.7e-7, which is the size of the failure. A CPU simulation with torch's AdamW gives
  -12 units for that case.
- The frozen bound (`4 x` the HF controls' error `+ 1e-7`) comes from controls whose gradient
  happened to differ by less than that at this element. Their update error is one unit (5.96e-8).
- TP4 sums the MLP input gradient over four ranks instead of two, so its rounding differs there.
  TP2 shows the same tensor at 4 units (0.70x).

`diagnose_update_sensitivity.py` (this directory) checks this on the real tensors. For each run
and tensor, it recomputes AdamW's first step in FP64 from the run's own unclipped gradient and grad
norm, and compares the predicted update difference with the observed one.
- It is CPU only and reads tensors through `fixed_contract.read_tensor`, which checks their hashes.
- It writes nothing into the run directories.
- It changes no bound and no acceptance file. It is a diagnosis, not a qualification step.

## Commands

On the machine that holds round 1's reference and round 2's candidate tensors, from `<src>` at
`19b62661` (clean tree, as round 2). `<D>` is this directory in the handoff checkout,
`<ref>` = `results/reference/job-<JOBID>-attempt-0`, `<cand>` = `results/candidate/job-<JOBID>-attempt-0`,
`<out>` = `results/round2/diagnosis` on the results branch.

```bash
cd <src>
export PYTHONPATH=<src> PYTHONNOUSERSITE=1
# 1. Three tensors, five runs (a few minutes)
<py> <D>/diagnose_update_sensitivity.py --reference <ref>/actor-fp32-0 \
  --other <ref>/actor-fp32-1 --other <ref>/actor-fp32-impl \
  --other <cand>/megatron-fp32-actor-tp2 --other <cand>/megatron-fp32-actor-tp4 \
  --copy-to <out>/tensors --output <out>/update-sensitivity.json > <out>/update-sensitivity.log 2>&1
# 2. Every step-0 update tensor of the TP4 run (reads about 130 GB; under an hour)
<py> <D>/diagnose_update_sensitivity.py --reference <ref>/actor-fp32-0 \
  --other <cand>/megatron-fp32-actor-tp4 --scan \
  --output <out>/update-sensitivity-scan-tp4.json > <out>/update-sensitivity-scan-tp4.log 2>&1
git -C <src> status --porcelain --untracked-files=no      # still empty
```

## Expected if the hypothesis holds

- In `update-sensitivity.json`, under `runs.megatron-fp32-actor-tp4.tensors.model.layers.17.post_attention_layernorm.weight`:
  - `max_abs_observed` 7.749e-07;
  - the first `top` element has `clipped_gradient_over_eps` below about 1;
  - its `update_difference` is close to `predicted_update_difference`;
  - `residual` is at the scale of the weights' FP32 rounding (at most about 1.2e-07).
- The same holds for TP2 and the two HF controls on this tensor, with smaller differences.
- In the scan, `max_residual` stays at rounding scale on every step-0 update tensor. In other
  words, every update difference of the TP4 run is explained by its gradient difference.

If the residual of the failing element is large instead, the hypothesis is wrong and the TP4
optimizer step itself differs. Send it as is.

## Send back

Push on top of the results branch `handoff/h100-pr548-item4-results-20261119` (after
`23001bb1`), under `results/round2/diagnosis/`:
- both JSON files and both logs;
- `tensors/` (step-0 gradient, update and grad norm of the three tensors per run, 16 KB each, so
  `git add -f` them).

Redact as before. Do not rerun step 6, open a PR or post comments.
