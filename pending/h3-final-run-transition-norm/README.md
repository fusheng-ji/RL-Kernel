# final_distributed_h3_run PR A: exact gradient norm, verl-omni H3 transition drop-in, NVRTC (2026-11-21)

Branch `feat/h3-final-run-transition-norm` at `164a5ebd` (stacked on #730 `23ea31cc`, on `test-h3` `12e81c36`), measured on one B200 node allocated to the job alone. Local paths are replaced by `<head>`, `<base>`, `<env>`, `<venv>`, `<verl-omni>` and `<python>`.

| Path | What |
|---|---|
| `pra-*.png`, `plot_pra.py` | the four charts of the PR's Results, drawn by `plot_pra.py` from the reports below |
| `verl-omni-compare/` | `h3_sample_previous_step` against verl-omni `a634f4d8`'s `sample_h3_transition` (CPS), 560 cases; torch 2.14.0+cu130, Diffusers 0.43.0 |
| `logprob-batch/` | a sample's log-prob scored in batches of 2 / 4 / 8 against alone, verl-omni and RL-Kernel, H3 latent sizes |
| `norm-prior-art/` | torch's norm and `exact_norm` over 16 single-process partitions / orders of the same LoRA-shaped gradients; torch 2.13.0+cu130 |
| `multirank/` | `torchrun` at 2 / 4 / 8 GPUs: TP2, TP4, TP8, TP4 x CP2; torch DTensor `clip_grad_norm_` and `clip_grad_norm_exact_` against the one-GPU clip |
| `megatron-hook/` | the Megatron `get_grad_norm_fp32` hook of `test-h3` against this PR's on the same gradients |
| `timing/` | cost per call on one idle B200: norm, clip, transition (video, audio; sampling, scoring) |
| `tests/suite-*.txt`, `failed-*.txt` | `pytest tests/models/minimax_h3 tests/ops/norm`, head on GPUs 0,1 and #730 `23ea31cc` on GPUs 2,3 at the same time, repo environment (`commits.txt`, `gpus.txt`) |
| `tests/nvrtc-cu13-*.txt` | `pytest tests/ops/norm/test_reproducible_norm.py` with only the CUDA 13 wheel NVRTC, at #730 `23ea31cc` and at the head |
| `tests/runtime-integration-transition-venv.txt` | `test_h3_runtime_integration.py::TestTransition` with verl-omni importable |
