# audio_vae_contract on H100 (2026-11-18)

- Code: `handoff/h3-h100-audio-vae-20261118` @ `a1177ddd` (detached worktree, `git status --porcelain --untracked-files=no` empty before and after). No patch applied. No `_C` build.
- Hardware: 1 x NVIDIA H100 (95830 MiB), driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, python 3.12.15, Diffusers source `f53d5520` on `PYTHONPATH`. Weights: Hub `MiniMaxAI/MiniMax-H3@42ed227` `FL2VA/audio_vae/` (`model.safetensors` sha256 `37dddc2f...`, verified). `--sglang` not used (no SGLang env). The visual VAE run (task B) ran concurrently in a separate one-GPU Slurm step on the same node (physical device not recorded).
- Commands:
  - `RL_KERNEL_H3_AUDIO_VAE=<dir>/FL2VA/audio_vae PYTHONPATH=.:<diffusers>/src python -m pytest -q -rs tests/models/minimax_h3/test_h3_audio_vae.py`
  - `python benchmarks/models/benchmark_h3_audio_vae.py --checkpoint $RL_KERNEL_H3_AUDIO_VAE --diffusers-src <diffusers>/src --repeats 10 --plot --out h3-audio-vae-h100.json`
  - same with `--repeats 0 --out h3-audio-vae-h100-rerun.json`

## Results
| Item | Expected | Observed |
|---|---|---|
| pytest | 33 passed, 0 skipped | **33 passed, 0 failed, 0 skipped** |
| `ours` encode/decode `batch_invariant` | true | **true** (all 8 entries) |
| all `max_abs` of `ours` (vs Diffusers / repeat / batch) | 0.0 | **0.0** (34 entries; the `vs_fp64_max_abs` fields are not part of this check and are about 1e-5 to 1e-4) |
| `cold_start.ours` | 0.0 | **0.0** (encode and decode) |
| `rows_sha256` and `paths.ours.{encode,decode}.sha256`, run 1 vs rerun | equal | **equal** (35 sha256 fields compared, 0 differ) |
| H100 hashes equal B200 hashes | not expected | not compared |

## Note on a first attempt
The first launch of this task (`attempt1-cuda-invisible/`) set `CUDA_VISIBLE_DEVICES=1` inside a one-GPU step, so CUDA was invisible: pytest gave 30 passed / 3 skipped ("needs CUDA") and the benchmark exited 2. That was a launcher error, not a test result; it is kept unedited. The results above are from the second launch with no `CUDA_VISIBLE_DEVICES` override.

## Draft "H100" paragraph
On one H100 (driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, Diffusers `f53d5520`, weights `42ed227` `FL2VA/audio_vae`), at `6d9c4ec3` code (handoff `a1177ddd`): `test_h3_audio_vae.py` 33 passed / 0 failed / 0 skipped; `ours` encode and decode are batch invariant with every `max_abs` 0.0 and cold start 0.0, and the row and path sha256 values are identical between two separate processes. No patch applied.
