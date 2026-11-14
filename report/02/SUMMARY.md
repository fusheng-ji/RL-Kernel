# 02. Bind `time_embedder` and `norm_out` to their rows in strict mode

Hardware: 1 H100 (94 GB) per run, torch 2.13.0+cu130, triton 3.7.1, provider `f53d5520`. Hosts redacted. Nothing pushed.

## Tree

- Branch `feat/h3-strict-time-embedder-norm-out`: `e87a15e4` (the binding, on `fix/h3-merged-rows-review-v2`) + merge
  `89cbd77f` of `fix/h3-strict-video-projection-session` (`a6c93313`, the session fix from item 01). **Tested SHA:
  `89cbd77f`.** The merge was needed because strict `fulltrans` does not run without it (item 01, "baseline failure");
  it conflicted in `h3_full_model.py` and its test file, resolved by hand: the keyed-row / session code of `a6c93313`
  and the `_NON_LINEAR_BOUNDARIES` set of `e87a15e4` are both kept, `PROVIDER_BOUNDARIES` is gone.
- Diff stat of `e87a15e4` against `fix/h3-merged-rows-review-v2` (5 files, +240 / -38):

```
 docs/operators/h3-full-transformer.md              |   9 +-
 rl_engine/validation/models/h3_full_model.py       |  63 +++++--
 tests/models/minimax_h3/test_h3_full_model.py      | 197 +++++++++++++++++++--
 .../minimax_h3/test_h3_runtime_integration.py      |   2 +-
 tools/validation/models/h3_full_transformer.py     |   7 +-
```

What it does: `time_embedder` (`timestep_mlp_fp32`) and `norm_out` (`final_adaln_out`) move into `STRICT_BOUNDARY_ROWS`
(`PROVIDER_BOUNDARIES` removed). `time_embedder` refuses a `condition`, a `cond_proj` and a `post_act`; `norm_out` passes
`(x, norm.weight, temb, linear.weight, linear.bias, timestep_indices, norm.eps)`; both honour `native` and are restored
on exit; they are kept out of the linear-projection loop. Only `time_proj` stays on the provider. The provider
signatures were checked against `f53d5520`.
The runtime-integration test relied on `boundary:norm_out` being an invalid overlay entry; it now uses `boundary:time_proj`.

## Tests and mypy

| Tree | SHA | passed | skipped | failed |
| --- | --- | --- | --- | --- |
| item 01 `followup` | `e3c4cbc9` | 1114 | 3 | 0 |
| `bind02` | `89cbd77f` | 1118 | 3 | 0 |

Same three skips as item 01 (`aenum` x2, `sglang`). CI-style mypy (no torch): clean (665 files). With torch: 36 errors in
12 files, the same set as the baseline.

## `fulltrans` (item 01's strict row is the reference)

| Run | Tree | strict | cand. / prov. repeatable | video rel | audio rel | input grad (video / audio / encoder) | param grad median / max | peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| item 01, strict | `a6c93313` | yes | yes / yes | 2.151e-03 | 2.104e-02 | 0.226 / 0.230 / 0.345 | 0.266 / 0.696 | 68.2 GiB |
| **item 02, strict** | `89cbd77f` | yes | **yes** / yes | 2.511e-03 | 1.543e-02 | 0.118 / 0.156 / 0.248 | 0.176 / 0.434 | 68.2 GiB |
| item 02, `--provider-boundaries` | `89cbd77f` | no | yes / yes | 2.211e-03 | 2.053e-02 | 0.142 / 0.146 / 0.247 | 0.282 / 0.924 | 68.2 GiB |

- `boundary_bindings` (strict): `proj_in`, `proj_out`, `audio_proj_in`, `audio_proj_out`, `context_embedder`,
  `token_refiner`, `time_embedder`, `norm_out`: **all `row`**.
- `--provider-boundaries` still leaves every boundary module on the provider (its `boundary_bindings` is empty), and its
  numbers are identical to item 01's provider-boundaries run: the B200 path is untouched.
- As expected, the strict output and input-gradient columns move (`temb` and the final norm change arithmetic): video
  rel 2.151e-03 -> 2.511e-03, audio rel 2.104e-02 -> 1.543e-02, input grads 0.226 / 0.230 / 0.345 -> 0.118 / 0.156 /
  0.248, param grads 0.266 / 0.696 -> 0.176 / 0.434. No committed bound on these columns.
- Candidate time 25.0 s (item 01 strict: 25.3 s).

## `dual` (H-C2), `89cbd77f`

| Seed | Provider | Strict |
|---|---|---|
| 11 | 16/16 differing | **0/16** |
| 22 | 16/16 differing | **0/16** |
| 33 | 16/16 differing | **0/16** |

Strict stays at item 01's result.

## Deviations

- Tested tree includes the item-01 session fix by merge (needed for strict `fulltrans`); that fix is a decision for
  Wenbo, see `../01/SUMMARY.md`. If it is dropped, this item's strict `fulltrans` needs another way to open the session.
