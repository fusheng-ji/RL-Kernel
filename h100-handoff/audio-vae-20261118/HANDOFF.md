# audio_vae_contract — H100 (Hopper) run for Wenbo

Nothing in this row *needs* Hopper: the audio VAE is a separate FP32 module (no H3 `_C`
kernels, no strict full-model run). Hopper is still worth one run, because the strict
contract leans on cuDNN/cuBLAS choosing the same algorithm for the same shape, and
those choices are per architecture. The B200 results are in `b200-reference/`.

## What to run (one H100, ~15 min)

This branch, `handoff/h3-h100-audio-vae-20261118`, is `feat/h3-audio-vae-contract` @
`6d9c4ec3` (base `10552d82`, `test-h3`) plus one commit that adds this folder. The code is
unchanged since `8c1d07b3` and the tests since `6bd47b7a`; the later commits are docs.

```bash
git fetch https://github.com/fusheng-ji/RL-Kernel handoff/h3-h100-audio-vae-20261118
git worktree add --detach <SRC> FETCH_HEAD && cd <SRC>
```

Inputs:
- released weights: Hub `MiniMaxAI/MiniMax-H3` @ `42ed227e`, `FL2VA/audio_vae/`
  (605 MB; `model.safetensors` sha256 `37dddc2f…5ea2`, checked by the code);
- Diffusers source at `f53d5520` (`git archive f53d55203 src | tar -x`), on `PYTHONPATH`
  or via `--diffusers-src`;
- SGLang 0.5.21 env for `--sglang` (optional; drop the flag if the H100 env has none).

```bash
export RL_KERNEL_H3_AUDIO_VAE=<dir>/FL2VA/audio_vae
PYTHONPATH=.:<diffusers>/src python -m pytest -q -rs tests/models/minimax_h3/test_h3_audio_vae.py
python benchmarks/models/benchmark_h3_audio_vae.py --checkpoint $RL_KERNEL_H3_AUDIO_VAE \
  --diffusers-src <diffusers>/src [--sglang] --repeats 10 --plot --out h3-audio-vae-h100.json
python benchmarks/models/benchmark_h3_audio_vae.py --checkpoint $RL_KERNEL_H3_AUDIO_VAE \
  --diffusers-src <diffusers>/src [--sglang] --repeats 0 --out h3-audio-vae-h100-rerun.json
```

## What to check

1. pytest: 33 passed, 0 skipped (with weights + diffusers present).
2. In `h3-audio-vae-h100.json`, every case: `paths.ours.encode.batch_invariant` and
   `paths.ours.decode.batch_invariant` are `true`, all `max_abs` of `ours` are `0.0`,
   `cold_start.ours` both `0.0`.
3. Cross-process: `cases[*].rows_sha256`, `paths.ours.{encode,decode}.sha256` equal
   between the two JSON files.
4. Expected *not* to hold: H100 hashes equal to the B200 hashes (different cuDNN/cuBLAS
   kernels per arch); the contract claims per-platform reproducibility, not cross-arch bytes.

If (2) fails on Hopper, batch composition cannot be the cause (micro-batch 1), so look at
repeat equality (`max_abs.repeat`) and `cold_start` first. A non-zero value there means the
deterministic cuDNN decode is not stable on SM90, and the decode policy should switch to
`cudnn_enabled=False` (measured 4x slower on B200). Send the JSONs back; the PR body has an H100 row reserved.

Push them, with the pytest output and `nvidia-smi --query-gpu=name,driver_version --format=csv`,
as fork branch `handoff/h3-h100-audio-vae-results-20261118` (folder
`h100-results/audio-vae-20261118/`), with host names and paths stripped. No PR and no comment.
