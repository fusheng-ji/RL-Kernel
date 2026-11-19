# phase_b_full_chain strict (R/R) on H100 (2026-11-18 handoff)

H100 (cc 9.0), 95830 MiB per card, driver in `nvidia-smi.csv`; torch 2.13.0+cu130, transformers 5.17.0.
Handoff branch `7e22c736`, nothing changed. Strict runs used `--offload --attention-query-chunk 4096` (see item 0).
`first_drift` is given as stage, step, module, row.

| Item | Result | first_drift | `gate.hard_gate` | `transformer_intermediates_equal` | Decoded media equal (audio/video) | `case_id` / `bound` / `fell_back` | Peak GPU memory |
|---|---|---|---|---|---|---|---|
| 0 env | cc (9,0), `_C` inside the worktree, `h3_full_attention False` | n/a | n/a | n/a | n/a | n/a | n/a |
| 0 tests | **2 failed, 38 passed, 0 skipped** (expected 40 passed) | n/a | n/a | n/a | n/a | n/a | n/a |
| 1 FL2VA A vs B (separate processes) | as expected | null | pass | true | true / true | R/R, true, null | 71.1 GB |
| 1 FL2VA A vs C (inject block 25, row 3000) | localized | denoise, step 1, `transformer_blocks.25`, row 3000 | fail | false | false / false | R/R, true, null | 71.1 GB |
| 2 Ref2VA A vs B (separate processes) | as expected | null | pass | true | true / true | R/R, true, null | 95.3 GB (card full) |
| 2 Ref2VA A vs C (inject block 25, row 20000) | localized | denoise, step 1, `transformer_blocks.25`, row 20000 | fail | false | false / false | R/R, true, null | 95.3 GB (card full) |
| 3 FL2VA strict vs provider | informational | denoise, step 0, `token_refiner` | fail | false | false / false | provider run P completed | 71.3 GB (provider) |
| 3 Ref2VA strict vs provider | informational | denoise, step 0, `token_refiner` | fail | false | false / false | provider run P completed | 71.3 GB (provider) |
| 4 FL2VA H100 provider vs B200 | informational | `visual_vae_encode`, `condition_latents.0` | fail | false | false / false | n/a | n/a |
| 4 Ref2VA H100 provider vs B200 | informational | `visual_vae_encode`, `condition_latents.0` | fail | false | false / false | n/a | n/a |

The `boundary_bindings` of both strict A runs have 8 `row` entries. Peak memory comes from `nvidia-smi` samples
taken every 2 s on the card used (`mem/`); the provider file covers both provider runs.

## Details

- **Item 1 A vs C.** `transformer.differing_rows = 1`, `first_row = 3000`, target video, latent frame 18, patch (6, 3),
  `position_thw = [359.0, 22.047..., 3.905...]`, the same row as on B200. The change was not rounded away: the
  trajectory differs from step 1 (video velocity), 8 of 12 compared entries differ, final latent differs.
- **Item 2 A vs C.** `differing_rows = 1`, `first_row = 20000`, role `condition`, `condition_row = 11683`, the same as
  on B200. Unlike B200, the trajectory differs from step 1 (8 of 12 entries), the hard gate is `fail`
  (on B200 it was `trajectory = null` and `pass`).
- **Item 2 A vs B.** No stage before `denoise` differs; all stages equal.
- **Item 3.** Every stage before `denoise` is equal; the two arms first differ at step 0 in `token_refiner` for both
  workflows. All 12 trajectory entries differ. Plots are in `plots/`.
- **Item 4.** Request to text conditions are equal, first drift in `visual_vae_encode`, as predicted.

## Deviations and things to know

- **`h3_full_attention` is False** (`_C` is a plain `setup.py build_ext --inplace`; the H3 attention needs a
  `RL_KERNEL_H3_STRICT=1` build and the handoff does not ask for that). I used the documented fallback
  `--attention-query-chunk 4096` for all strict runs (`actual_backends.attention` is `QueryChunkAttention`).
- **Tests 2 failed.** `test_sealed_fixture_replays[fl2va]` and `[ref2va]`: one asserts at
  `media_preprocess`, `keyframe_pixels.0`, the other at `audio_vae_encode`, `audio_rows.0` (see `00/tests.txt`).
  Not investigated further.
- **FL2VA run C was run twice.** The first attempt failed writing its output (disk quota, not a computation error).
  `01/status.txt` keeps both lines; the results above are from the successful rerun (A, B unchanged).
- **Ref2VA** used `--offload` (no `--stream-blocks` needed): the card was at 95.3 GB, with allocator OOM warnings
  in `02/ref2va-A.log`, but all runs exited 0. Runs B and C were run at the same time on two cards, each in its own
  process; A ran alone first. The sequential driver was stopped after A.
- Checkpoint dir: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`, downloaded with Hub metadata
  (`transformer_ref/` re-downloaded; shard 10 of it was completed after a quota error).
- `artifact_sha256` fields inside `*-report.json` were taken before redaction. Redaction replaced host names,
  paths, user and job id with `{...}` placeholders, so a few redacted JSON/log files will not hash to those values.
- `*.safetensors` and `tmp/` are not included.

## 05: text feature check (ADDENDUM, branch `97f7f819`, separate worktree)

Contract run on the same H100 node, CPU and CUDA, against the pinned features (`05/*.json`).

| Workflow | Device | `same_conditioner_input` | `tags_equal` | `bitwise_equal` | `max_abs_diff` | `relative_l2` | differing tokens (text / vision / total) | first differing token |
|---|---|---|---|---|---|---|---|---|
| FL2VA | cpu | true | true | **false** (expected true) | 256 | 0.01893 | 31 / 228 / 259 | 0 |
| FL2VA | cuda | true | true | false | 320 | 0.02648 | 31 / 228 / 259 | 0 |
| Ref2VA | cpu | true | true | **false** (expected true) | 14287 | 0.31971 | 51 / 8166 / 8217 | 0 |
| Ref2VA | cuda | true | true | false | 13583 | 0.36465 | 51 / 8166 / 8217 | 0 |

On this host the contract on CPU does **not** reproduce the pinned bytes (B200's CPU did). B200 CUDA, for
comparison from the ADDENDUM: FL2VA 0.0380 / 832, Ref2VA 0.4615 / 16063.
