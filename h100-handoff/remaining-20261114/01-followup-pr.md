# 01. The follow-up PR for the merged rows (#568 / #570 / #571 / #574)

Branch `fix/h3-merged-rows-review` @ `0264a4f8` (base `cfec0d98`). Six signed commits, no AI trailers:

| Commit | Fix |
|---|---|
| `ff395298` | `OperatorCall.fallback` defaults to `None` (not observed); `None` survives `dumps` / `loads` |
| `25e2b8d2` | projection rows: fail-closed CUDA dispatch, fused `[Q; K; V]` backward in the block graph, reference rows take the CUDA signature (`logical_keys`), all-padding call gives zero parameter gradients |
| `1a118e81` | the token refiner's q/k/v nodes share one fused QKV call |
| `7033fbfc` | `H3DiffusersAdapter.active()` counts only a failure to bind as an audit fallback |
| `e3b386ec` | `fixed_order_mean` is an `autograd.Function`, so the CPS `log_prob` carries the policy gradient |
| `0264a4f8` | `lora_gradient_gate`'s `base_frozen` is a real check; `transition()` refuses mismatched `sample_ids`; doc gives the `VERL_OMNI_PATH` condition |

## Step 1. Merge-up and overlap check (no GPU)

```bash
git switch -c fix/h3-merged-rows-review-v2 fix/h3-merged-rows-review
git merge --no-edit upstream/test-h3        # 07533aee; merge-tree found no conflict
git worktree add --detach <SRC>/followup HEAD
```

Since `cfec0d98`, `test-h3` changed two of the files that this branch changes:
- `rl_engine/validation/models/h3_block.py`: `4f7c5d11` binds the `h3_full_attention` row (#553) in the strict block and
  times the full model. Check that the fused QKV node of `25e2b8d2` / `1a118e81` still feeds the attention node it binds,
  in forward and backward.
- `rl_engine/runtime/registry.py`: `451311bb` (softcap) and `5e80fe12` (MUSA GEMM) touch other rows. Check that the
  fail-closed CUDA priorities of `25e2b8d2` are still the only entries for `audio_input_projection_fp32`,
  `audio_output_projection_fp32`, `text_context_projection` and `h3_qkv_gemm`.

Also check that the #569 / #575 fixes that landed in `test-h3` while they were open do not already do one of the six
fixes. A duplicate has to be dropped from this branch, not kept twice:
`39cb7d3d` (explicit native QKV layout, exact adapter restore), `100057f7` (biased projection refused in the rollout
adapter), `cca6e2fb` / `61619baf` (short CP shards), `9188133b` / `e2b09db7` (hierarchical subgroups). Compare with
`git show <sha> --stat` and the diff of each of the six commits.

## Step 2. Test suite and mypy (one node, 4 GPUs visible)

```bash
scripts/run_h100.sh tests <SRC>/base     <OUT>/01/tests-base       # 07533aee, the baseline
scripts/run_h100.sh tests <SRC>/followup <OUT>/01/tests-followup
scripts/run_h100.sh mypy  <SRC>/followup <OUT>/01/mypy
```

**Expected:**
- no failure in `followup` that `base` does not also have;
- `mypy` clean.

These tests must run with the weights, not skip:
- `test_h3_one_block.py` and `test_h3_token_refiner.py`: the block and refiner graphs now use the fused QKV backward. Their
  committed accuracy bounds against FP64 stay as they are.
- `test_h3_runtime_integration.py`: includes the verl-omni scheduler comparison when `VERL_OMNI` is set.

Send back passed / skipped / failed for both trees, the tracebacks of the failures that are not in the baseline, and the
skip reasons.

## Step 3. 33B strict vs provider, and rollout vs train (RFC H-C2), one H100 each

```bash
scripts/run_h100.sh fulltrans <SRC>/followup <OUT>/01/fulltrans
scripts/run_h100.sh dual      <SRC>/followup <OUT>/01/dual          # seeds 11 22 33
```

`h3_full_transformer.py` takes no seed: `fulltrans` runs it once in strict mode, and once with `--provider-boundaries`
(boundary modules on the provider, as the B200 comparison does). For each run, record:
- candidate bitwise repeatable, provider repeatable;
- video and audio output relative difference;
- input-gradient relative differences (video hidden states, audio, encoder);
- parameter-gradient median and maximum;
- peak memory.

The last H100 numbers in #571 (head `5ae9f5eb`) were taken before the video projections were bound and before the fused
QKV backward: video 2.07e-03, audio 1.77e-02, input gradients 0.23 / 0.23 / 0.33. The output differences are expected to
move (strict `proj_in` / `proj_out`), and so are the input gradients (fused QKV backward). There is no committed bound on
these columns, so report the numbers. A candidate that is **not** repeatable is a failure.

`h3_dual_trajectory.py`, the #571 H100 table from before the binding:

| Seed | Provider | Strict | Strict + video projections (old diagnostic, now removed) |
|---|---|---|---|
| 11 | 16/16 | 8/16 | 0/16 |
| 22 | 16/16 | 8/16 | 0/16 |
| 33 | 16/16 | 8/16 | 0/16 |

The diagnostic bound the video projections to the generic `h3_projection` GEMM. Strict mode now binds the dedicated
rows of #646, which are a different kernel.

**Expected (H-C2):** strict 0/16 on all three seeds, provider 16/16. If strict is not 0/16, that is a finding, not
something to tune. Record the first differing step and tensor, the max abs difference per step, and whether video,
audio or both differ.

## Step 4. PR text (no GPU)

Update `followup-pr-draft.md`:
- it says "#569 and #575 stay open"; both are merged now;
- base `cfec0d98` → the merge-up SHA;
- add the step 2–3 results (What → Prior art → Results → Status → Tests → Scope);
- cite only public commits, and link evidence on `pr_assets` once it is uploaded.

Wenbo pushes and opens the PR himself.

## Send back

`tests-base/`, `tests-followup/`, `mypy/`, `fulltrans/`, `dual/`, the merge-up SHA, the overlap findings of step 1, and
`SUMMARY.md` with the two tables above filled with this run's numbers.
