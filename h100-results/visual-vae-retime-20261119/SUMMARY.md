# visual_vae_contract decode re-time on H100 at `0028ad7d` (handoff `bbdddc67`)

Result: **as expected, `RESULT: PASS`.** The outputs did not change and the batch-1 decode is about 1.9 times faster than
at `1e1ff922`.

| Check | Expected | Observed |
|---|---|---|
| pytest | 221 passed | 221 passed in 43 s |
| evidence exit code | 0 | 0 |
| required verdicts | all 7 true | all true |
| `diffusers_batched_batch_invariant_default_flags` (informational) | false | false |
| `diffusers_parity_default_flags` (informational) | true | true |
| sha256 fields differing from the first H100 run | 0 of 30 | 0 of 30 |
| `compare.txt` | ends in `RESULT: PASS` | `RESULT: PASS` |

Median timings in ms (first run at `1e1ff922` against this run):

| Case | `1e1ff922` | this run |
|---|---|---|
| contract / decode_video | 1126.3 | **602.0** |
| contract_untiled / decode_video | 988.7 | 469.4 |
| diffusers_default_flags / decode_video | 491.2 | 491.3 |
| diffusers_pinned_flags / decode_video | 491.0 | 490.3 |
| contract / encode_video | 1849.0 | 1852.1 |
| diffusers_default_flags / encode_video | 1847.1 | 1853.7 |
| contract / encode_keyframe | 97.4 | 97.5 |
| contract_untiled / encode_keyframe | 41.9 | 56.9 |

The Diffusers decode stayed at about 491 ms, so the card was in the same state as in the first run. The contract
decode is 602 ms against 491 ms for Diffusers (B200: 402.7 against 339.1 ms). `contract_untiled / encode_keyframe`
is slower here (56.9 against 41.9 ms); nothing else changed by more than a few ms.

## Notes

- The card was idle before the run: `gpu-before.txt` shows 0 MiB used and 0 % utilization. NVIDIA H100, one card
  (95830 MiB), driver 580.173.02, torch 2.13.0+cu130, CUDA 13.0, Diffusers `f53d552036a0d1bd5570782a39cd40cfabf112bc`.
- **`rl_kernel_dirty` is true** in the evidence (and `compare.txt` prints `dirty: True`). `git status` on the
  worktree lists 21 tracked files; `git diff --summary` shows they are all `mode change 100755 => 100644`, with 0
  insertions and 0 deletions. With `core.fileMode=false` the status is empty. These are mode-only changes from the
  checkout on this shared file system (the same 21 files as in the qwen3vl r2 checkout); no content changed.
- `evidence-h100.json` was staged unchanged (byte-identical to the original after redaction found nothing to
  replace). Host names, paths and the job id are otherwise replaced with `{...}` placeholders; hashes are untouched.
