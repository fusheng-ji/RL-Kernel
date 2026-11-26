# H100 results: MiniMax-H3-Image rows of RFC #738 (handoff 44a2b97b5; T1–T4 run from 1a7c454c3, whose T1–T4 text is unchanged)

Hardware: NVIDIA H100 (95830 MiB, driver 580.173.02), GPU 0 of an otherwise idle node for every task:
- T1 on node-a;
- T3, T4 and T2 on node-b (one after another);
- T5 on node-c.

Every node showed 0 MiB used at the start of each task (`env.txt`). torch 2.13.0+cu130, CUDA 13.0, Python 3.12.15.
Environment:
- `<V>` was built exactly as in the README: `<E1>/bin/python -m venv --system-site-packages <V>`, then `pip install --no-deps` of Diffusers `875373ac`, which gives `0.42.0.dev0` (`pip-list.txt`).
- Because `<E1>` is itself a venv, `<V>` inherits E1's conda base environment, not E1's own layer:
  - `<V>` has transformers 5.17.0 (like B200), Pillow 13.0.0 (B200 12.3.0) and tokenizers 0.23.3 (B200 0.23.2);
  - E1's own layer has Diffusers 0.41.0 and transformers 5.13.1. This matters for T1 b and d.
- Weights:
  - `<IMG>` is the README `snapshot_download` at `8ced6006`; the VAE sha256 is `5bb28096…`, as required;
  - `<CKPT>` is the MiniMax-H3 checkpoint at `42ed227e`.
- Code:
  - both bundles verified;
  - all 11 refs match `refs.txt`;
  - upstream `test-h3` is now `fc4989f7` and contains `3f93cae8b`; `<B>` is `3f93cae8b` as written.
- Every worktree's tracked status was empty before and after each step.

| Task | Command (README) | Exit | Headline |
|---|---|---|---|
| T1 build | `setup.py build_ext --inplace` in `<W>` and `<B>` (E1) | 0 / 0 | `_C` inside each checkout (`T1/c-file-*.txt`) |
| T1 a | pytest `tests/models/minimax_h3_image`, `<V>` | 0 | **320 passed, 0 skipped**, as on B200 |
| T1 b | the same pytest with `<E1>` | 1 | **3 failed**, 267 passed, 50 skipped (B200: 264 passed, 56 skipped); see below |
| T1 c | `h3_image_fingerprint.py check` | 0 | `ok: 0eefe73a8ffa69ac3a4cf8bebcf3b770d3bbf48c21320df1fe72f2ec5ad02579` |
| T1 d | pytest `tests/models` at `<W>` and `<B>` (E1, `RL_KERNEL_H3_WEIGHTS` set) | 1 / 1 | head: 4 failed, 1694 passed, 207 skipped; base: 1 failed, 1427 passed, 157 skipped. Fails only at head: the same 3 tests as T1 b |
| T2 | `h3_image_cross_env_probe.py --compare probe-b200-89061319.json` | **1** | `hardware-independent digests differing: 1` (`canvas.plans`); Pillow resize digests differing: 1; all 30 `cuda_equals_cpu` / `cuda_round_trip` True; drift `repeat_bitwise_equal` true, max_abs fp64 4.4e-16 (B200 3.3e-16), fp32 2.4e-7 (B200 2.4e-7) |
| T3 pytest | pytest `tests/models/minimax_h3_image` at `<W3>` | 0 | 92 passed |
| T3 evidence | `h3_image_vae_evidence.py --verify-sha256 --timing-repeats 7`, then `--check-manifest` | 0 / 0 | all verdicts equal B200's (only `diffusers_batched_batch_invariant_default_flags` false); tiled decode 2150 ms (B200 1486), untiled 1168 ms (860); encode tiled 344 ms (264), untiled 141 ms (103); peak 9.37 GiB (9.36); `manifest up to date` |
| T4 evidence | `h3_image_text_presentation.py evidence --verify-weights` | 0 | `all_equal_diffusers` True (full model and 50-layer stack, all 4 cases), `all_cpu_digests_match_sealed` True, `--verify-weights` checked 23 text-encoder files against the pin `h3-qwen3vl-layer50-v1` without error, stack peak 50.6 GiB (B200 50.59) |
| T4 pytest | pytest `tests/models/minimax_h3_image` at `<W4>` | 0 | 95 passed, 0 skipped (`test_release_fixtures_bitwise_against_diffusers` ran and passed) |
| T5 build | `setup.py build_ext --inplace` in `<W5>` (E1) | 0 | `ext.txt`: path inside `<W5>`, `True` |
| T5 a | pytest `test_h3_image_reference_conditioning.py` | 0 | **46 passed, 0 skipped** |
| T5 b | `h3_image_reference_conditioning.py --device cuda` | 0 | `arms not equal to the CPU reference or failing the fixed-row loop: none`; CPU-generator arms equal B200 for all 4 fixtures; `probes` equal B200 exactly |
| T5 c | `h3_image_refcond_real_vae_check.py` | 0 | `all_green True`; all 4 arms bitwise equal to Diffusers, rows fixed over 6 steps |
| T6 | `h3_image_full_chain.py --root <IMG> --text-encoder-root <CKPT> --out <OUT>/T6/run --repeat`, then pytest `test_h3_image_full_chain.py` (`<V>` + accelerate 1.16.0) | 0 / 0 | `exit=0`, `all_gates_pass True`; all 4 fixtures: `gate`, `secondary_equal`, `run_to_run_equal` true, `first_drift` null, all 11 stages equal; **26 passed**; peak memory 64.9 / 67.0 / 67.0 / **67.3** GiB (`t2i_upscaled` / `t2i_example` / `edit_two_refs` / `edit_example`); seconds per fixture, RL-Kernel chain / Diffusers: 4.6 / 6.4, 21.7 / 28.8, 17.3 / 19.1, 23.0 / 29.8 |

## Findings

**T1 b (not green).** The 3 failures are:
- `test_h3_image_packed_layout.py::test_consistent_permutation_preserves_the_transformer_output`
- `test_h3_image_packed_layout.py::test_partial_permutation_changes_the_transformer_output`
- `test_h3_image_trajectory.py::test_loop_matches_pinned_scheduler_and_timestep_plan`

All three fail with `TypeError: MiniMaxH3Scheduler.__init__() got an unexpected keyword argument 'shift_provided_sigmas'`.
- `<E1>` is not a Diffusers-free environment: its own layer has Diffusers 0.41.0, which ships an older `MiniMaxH3Scheduler`.
- So these tests are not skipped as in the B200 CI view; they run against the old scheduler and fail.
- T1 d fails at head only on these same 3 tests, since they do not exist at the base, for the same reason.
- The other failure, `tests/models/minimax_h3/test_h3_report_timing.py::test_the_reported_medians_come_from_the_stored_samples`, fails at both head and base.
- The files are under `T1/` and, unchanged, under `not-green-attempts/T1-b-and-d-e1-diffusers-0.41/`.

**T2 (exit 1).** The one hardware-independent digest that differs is `canvas.plans`.
- The probe hashes `resolve_canvas(...).artifact()["sha256"]` for each of the 46,117 plans.
- `artifact()` in `rl_engine/models/minimax_h3_image/canvas.py` puts `"pillow": _pillow_version()` into the hashed body whenever the plan needs a resize.
- So this digest depends on the Pillow version (13.0.0 here, 12.3.0 on B200), although the probe lists only `canvas.resize` as Pillow-dependent.
- No other hardware-independent digest differs. The one differing resize digest (`48x1000@1024`) is the expected Pillow difference.

**T3 digests.** 4 of 14 output digests differ from B200: `decode.uint8`, `decode.unit`, and the two reference `sha256`. The other 10 match: the latent and parameter stats, the LUT, the tiling plans and the generator draws. Latency is 1.3–1.4x B200's.

**T5.**
- `rows_sha256` of the CUDA-generator arms equals B200 for `edit1`–`edit3` but not for `hires2` (reported only, as the README allows).
- Within H100, `cuda-cuda_generator` and `cuda_kernel-cuda_generator` are equal for all four fixtures.
- In c, `rows_sha256` differs from B200 in all 4 arms. This is reported only: it depends on the GPU's VAE convolutions.

## Notes on the harness (not on the results)
- `T1/rc.txt`: the `build …` and `d …` lines record `rc=$?` after a `$(basename …)` substitution in the same `echo`, so they show the substitution's status 0, not the command's. The exit codes in this table come from the logs: all builds completed, and both `tests/models` runs had failures, so pytest's exit code was 1.
- `T3/c-file-v.txt` and `T4/c-file-v.txt` say `inside`, but `<W3>`/`<W4>` are not built: `import rl_engine._C` raised `ModuleNotFoundError`, and the `inside` refers to the last printed line, `rl_engine/__init__.py` in the checkout. No `_C` was used there; the image rows do not need it.
- T2 was run on node-b while T1 was still running on node-a. The node-a script, which had T2 queued after T1, was cancelled right after `T1 end`, during T2's setup and before the probe started. Its partial files (nvidia-smi, commit, env, host) are in `not-green-attempts/T2-node-a-cancelled-before-probe/`.

**T6 (full chain on the released weights).**
- Everything the README expects holds: the exit code, the gates, the 26 tests, `first_drift` null at every stage, and the digests of the two arms agree stage by stage on the same GPU.
- Peak memory is within 0.2 GiB of B200's on every fixture. `edit_example` peaks at 67.3 GiB, 0.1 GiB above the README's "at most 67.2 GiB" (B200 measured 67.2); it fits on the 80 GB card.
- Time: the whole runner took about 10 minutes (B200: 8 minutes). Per fixture, the RL-Kernel chain takes 1.5–1.7x B200's time (4.6–23.0 s vs 3.0–14.6 s), and the Diffusers arm takes 1.5–1.7x B200's (6.4–29.8 s vs 4.2–18.5 s).
- `final_latent_sha256` and `output_sha256` differ from B200's for all four fixtures, as the README expects for BF16 matmuls (the comparison is made elsewhere).
- Setup: `accelerate==1.16.0` installed into `<V>` only (`<E1>` still sees 1.15.0); `transformer_diffusers/`, `examples/` and `assets/showcase/photo.jpg` were added to `<IMG>` (28 files, `T6/setup/files-added.txt`).
- Hosts: T6 ran on node-c, GPU 0, which showed 0 MiB and no other job at the start.
- The first launch of T6 on node-a was stopped within a minute of starting, because another job of the same allocation had just started on that node and the GPU was no longer guaranteed idle. Its partial records are in `not-green-attempts/T6-aborted-on-node-a/`; it produced no result.
- The 8 PNG files are not committed, as the README asks.

## T8 / T9 (2026-11-26)

T8 ran `b685c4991` (PDD head + one step), T9 ran `69e0e42c9` (ws1_full), each on one idle H100 of a shared node (`env.txt`). On H100 the FP32 boundary rows and the three GEMM rows bind as rows: bindings 30 row / 6 interim / 3 reference (B200 22 / 14 / 3). Logs are not uploaded; the JSON and pytest files are.

| Task | Exit | Headline |
|---|---|---|
| T8 `run-t2i` (`t2i_upscaled`, six steps) | 0 | `ok` true; controls `ok`: repeat, batch rows 0 / 1 and permutation first drift null, next head first drift at `pdd_proj_out`, training step ok |
| T8 `run-edit` (`edit_two_refs`) | 1 | the batch control batches 2 x 17,203 = 34,406 rows; `h3_ffn_down_gemm` refuses more than 32,768 (`row count must be in [1,32768]`). Fixed in `f4171b776`; not rerun yet |
| T8 pytest new / image suite | 1 / 1 | 82 passed, 1 failed / 513 passed, 1 failed: `test_cuda_projection_is_accurate_and_row_invariant`, `x must be contiguous` (the SM90 projection row on a slice). Fixed in `d3bc9fe41`; not rerun yet |
| T9-t2i `run-a` (`t2i_upscaled` controls + training at steps 0-5, `t2i_example`) | 0 | `ok` true; every control drift null; training ok at all six heads; schedule and inputs equal to Diffusers; strict chain 44.1 s / 579 s (B200 42.2 / 481), peak 65.3 / 69.2 GiB; PSNR to Diffusers default, strict / closest variant: 17.8 / 17.2 dB (`t2i_upscaled`), 18.2 / 17.7 dB (`t2i_example`) |
| T9-t2i `run-b` | 0 | `equal_to_other_process` true for both fixtures |
| T9-t2i pytest new / suite | 0 / 1 | 17 passed / 530 passed, 1 failed (the same projection test) |
| T9-edit | - | the job hit its time limit before the first fixture finished; to be rerun with the fixes |
