# 01. Follow-up PR for the merged rows (#568 / #570 / #571 / #574)

Hardware: 1 H100 (94 GB) per run, torch 2.13.0+cu130, triton 3.7.1, provider `f53d5520`, 4 GPUs visible for tests.
Hosts are redacted. Nothing was pushed.

## Merge-up

- Branch `fix/h3-merged-rows-review-v2` = `fix/h3-merged-rows-review` (`0264a4f8`) + merge of `upstream/test-h3`
  (`07533aee`). **Merge-up SHA: `e3c4cbc9`** (no conflict). The six commits are all still there; none was a duplicate.

## Step 1. Overlap check

- `h3_block.py` (`4f7c5d11`, binds `h3_full_attention`): the fused QKV node of `25e2b8d2` / `1a118e81` still feeds the
  attention node, in forward and backward. Evidence: `test_h3_one_block.py` (10 passed) and `test_h3_token_refiner.py`
  (16 passed), with their committed FP64 bounds unchanged.
- `registry.py` (`451311bb`, `5e80fe12`): they touch other rows. The CUDA priority lists of
  `audio_input_projection_fp32`, `audio_output_projection_fp32`, `text_context_projection` and `h3_qkv_gemm` still have
  the single fail-closed CUDA entry each (`25e2b8d2`); the PyTorch entries are only in the non-CUDA lists.
- #569 / #575 fixes that landed in `test-h3`: `39cb7d3d`, `100057f7` touch `h3_omni.py`, `checkpoint_mapper.py`,
  `h3_full_speed.py`; `cca6e2fb` / `61619baf`, `9188133b` / `e2b09db7` touch `parallel.py`, `hierarchical.py`,
  `h3_tp.py`, WS2 tools. None of the six commits touches those files, and none repeats one of their changes. Nothing
  to drop.

## Step 2. Tests and mypy

| Tree | SHA | passed | skipped | failed |
| --- | --- | --- | --- | --- |
| baseline `test-h3` (`base`) | `07533aee` | 1102 | 3 | 0 |
| `followup` | `e3c4cbc9` | 1114 | 3 | 0 |

- No failure in either tree. Skips are the same three: two `aenum` imports for `vllm_omni`
  (`test_h3_checkpoint_mapper_vllm.py:72`, `:215`) and one `sglang` import (`test_h3_timestep_sinusoid.py:129`).
- Required tests ran, none skipped: `test_h3_one_block.py` 10, `test_h3_token_refiner.py` 16,
  `test_h3_runtime_integration.py` 26 (with `VERL_OMNI` set; 19 at the baseline).
- mypy as CI (`--ignore-missing-imports rl_engine/`, no torch): **clean** at `07533aee` and `e3c4cbc9` (665 files).
  With torch installed, `mypy rl_engine` reports 36 errors in 12 files at **both** (same set, pre-existing, in
  `rl_engine/integrations/...`), so it is not a regression of this branch, and it is not what CI runs.
  Folders: `mypy-base/`, `mypy/` (`ci-mypy.log` is the CI-style run).

## Step 3. 33B strict vs provider, and rollout vs train (H-C2)

### A baseline failure (a finding)

`fulltrans` in strict mode **fails on `base` (`07533aee`) and on `followup` (`e3c4cbc9`)**:
`RuntimeError: parameter gradients require canonical_backward_session`, raised in
`rl_engine/ops/gemm/video_input_projection_fp32.py:113`. The strict video projection rows (#562 / #646) need a canonical
backward session, `logical_keys` and a `parameter_id` for a trainable weight, and the full-model harness provides none.
The provider-boundaries run is not affected. Logs: `fulltrans-base/`, `fulltrans/` (`full-transformer.log`).

To get the strict numbers I fixed it in a commit of its own, `a6c93313`
(`fix/h3-strict-video-projection-session`, on top of `e3c4cbc9`): the harness opens a
`canonical_backward_session` and passes `logical_keys` and `parameter_id` to the two keyed rows. The strict numbers
below are from that tree (`fulltrans-vsess/`). This fix is **not** in the `followup` branch and is a decision for
Wenbo (it is in item 02's branch by merge).

### `fulltrans` (one run each; the harness takes no seed)

Candidate = RL-Kernel rows, provider = diffusers. Both bitwise repeatable in both runs (a non-repeatable candidate would
be a failure; none is).

| Run | Tree | strict | cand. repeatable | prov. repeatable | video rel | audio rel | input grad (video / audio / encoder) | param grad median / max | peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #571 (before this work) | `5ae9f5eb` | | | | 2.07e-03 | 1.77e-02 | 0.23 / 0.23 / 0.33 | | |
| `--provider-boundaries` | `e3c4cbc9` | no | yes | yes | 2.211e-03 | 2.053e-02 | 0.142 / 0.146 / 0.247 | 0.282 / 0.924 | 68.2 GiB |
| strict | `a6c93313` | yes | yes | yes | 2.151e-03 | 2.104e-02 | 0.226 / 0.230 / 0.345 | 0.266 / 0.696 | 68.2 GiB |

- Baseline for the provider-boundaries column (`07533aee`): 2.211e-03 / 2.053e-02, input grads 0.143 / 0.146 / 0.244,
  param grads 0.279 / 0.930: the fused QKV backward moves the input gradients by 1e-3 to 3e-3 (absolute).
- Strict `boundary_bindings`: `proj_in`, `proj_out`, `audio_proj_in`, `audio_proj_out`, `context_embedder`,
  `token_refiner` = `row`; `time_embedder`, `norm_out` = `provider` (item 02 binds them).
- Candidate time: 12.2 s (provider-boundaries) vs 25.3 s (strict), because the strict rows are slower (candidate
  forward 1.1 s vs 7.1 s). Peak memory is the same, 68.2 GiB.
- Output differences are close to the pre-binding numbers; the strict input gradients are close to #571's, the
  provider-boundaries ones are lower. There is no committed bound on these columns.

### `dual` (rollout vs train, 4 Euler steps, 16 compared tensors per seed), `e3c4cbc9`

| Seed | Provider | Strict | Strict + video projections (old diagnostic, removed) |
|---|---|---|---|
| 11 | 16/16 differing | **0/16** | not run (removed) |
| 22 | 16/16 differing | **0/16** | not run (removed) |
| 33 | 16/16 differing | **0/16** | not run (removed) |

H-C2 holds as expected: strict 0/16 on all three seeds, provider 16/16. Provider first drift on every seed: step 0,
video velocity, 384 of 384 elements differ (max abs diff 0.4747 / 0.2547 / 0.5449 for seeds 11 / 22 / 33). Strict has no
drift. (#571's earlier table, before the binding, had strict at 8/16.)

## Step 4. PR text

`followup-pr-draft.md` in this folder's parent is updated: #569 and #575 are merged, the base is the merge-up SHA
`e3c4cbc9` (a local merge commit, it is not public: the PR gets a new SHA after Wenbo pushes), the H100 results are in,
and only commits that are public are cited. It is not pushed.

## Deviations

- `fulltrans` strict failed on the unchanged branch (above): the numbers come from the extra commit `a6c93313`.
- mypy with torch is not clean at the baseline either (36 errors); CI-style mypy is.
