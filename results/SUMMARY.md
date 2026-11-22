# PR #548 review item 4: Qwen3-8B fixed-rollout qualification on H100 (2026-11-19/20)

**Round 2 (2026-11-22, `19b62661`): see the "Round 2" section at the end. Unit tests 26 passed; step 5 probe: all 8 rows `probe_complete`, highest peak 86.0 GiB per rank. Step 6 candidate: run 1 (Megatron FP32 actor TP2) passed, run 2 (Megatron FP32 actor TP4) failed on one tensor of 1604 (`0/update/model.layers.17.post_attention_layernorm.weight`, 2.29 x its bound); the script stopped there (rc 1).**

**Status: steps 1-4 passed. Step 5 (TP2 memory probe) did NOT meet its go rule (5 of 8 rows `probe_complete`), so I stopped: step 6 (candidate, tiny-GPT regression) has not been run.** Commit `b6412e38` (steps 1-4, reference without tensors) was pushed before the probe started and shows that `acceptance.json` and all `*-bounds.json` were frozen before any probe or candidate run. The tensors (about 1.4 TB, scratch) stay on disk.

## Hardware and software
- One node, 4 x NVIDIA H100 (95830 MiB each, NVLink NV6 between all pairs, topology in `nvidia-smi.txt`), 755 GB host RAM (about 630 GB available during the reference), 64 CPU cores, driver 580.173.02, CUDA 13.0 (torch build), nothing else of mine on the node. The node belongs to a larger allocation; one other node of it was idle.
- trainer: Python 3.11.15, torch 2.9.1+cu130, transformers 5.13.1; overlay: megatron-core 0.15.3, deepspeed 0.19.7; vLLM environment: vllm 0.15.1 with torch 2.9.1+cu130. Built with `setup_envs.sh` (`results/steps/setup_envs.sh.txt`).
- Run code: detached worktree at `1bba6d193233a8279fb85203d26389b0e453712c` (`commit.txt` in every run; `git status --porcelain` empty before and after, see `results/steps/*status-after.txt` and `results/step1-2-build-and-smoke/`). `core.fileMode=false` was set through `GIT_CONFIG_*` variables because this filesystem reports file mode changes; no file was edited.
- Model: `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`, all 15 files checked against the manifest by the scripts.
- Every step ran with `PYTHONNOUSERSITE=1`, a short `TMPDIR` (`/tmp/q548-<pid>`), `ninja` on `PATH`, `TORCH_CUDA_ARCH_LIST=9.0` for the build. `results/steps/env.sh.txt` is the common environment.

## Deviations and environment fixes (none changes a parameter of the qualification)
1. `CUDA_HOME` had to be set for the build (a CUDA 13 toolkit directory); attempt 1 failed without it (`not-green-attempts/`, `step12-attempt1-no-CUDA_HOME`).
2. The local model directory lacked `.gitattributes`, `LICENSE`, `README.md`, so the manifest check failed in calibration attempt 1. I made a separate checkpoint directory with hard links to the same weight files and the three missing files downloaded from the Hub at the same revision; the scripts verify every file's sha256.
3. vLLM 0.15.1 (a CUDA 12 wheel) needs `libcudart.so.12`, which this node does not have; reference attempt 1 stopped in 78 s (`runtime.log`). I installed `nvidia-cuda-runtime-cu12==12.9.*` into a separate directory and added it to `LD_LIBRARY_PATH`; nothing was installed into the shared environments.
4. The project file system quota filled up at 637 GB (`torch.save` write failure in `actor-bf16-impl`, reference attempt 2, `not-green-attempts/attempt2-reference-disk-quota`; tensors of that attempt were deleted, the non-tensor files are kept). The reference and everything after it write to a scratch file system. This was not a numeric or memory failure.
5. The reference attempt that produced this evidence ran the unmodified `qualify_reference.sbatch` from the start (job-local run id `job-<JOBID>-attempt-0` in a new scratch directory), 2026-11-19T17:45:58Z to 2026-11-20T03:41:17Z, rc 0, 35719 s. I started a second script (`results/steps/resume_ref.sh.txt`) for a few minutes because I wrongly thought the first one had been killed; it started no GPU run and wrote no output (its first launch stopped on `compare_fixed` refusing to overwrite an existing file); I stopped it at 2026-11-19T21:53Z. The run of the original script was not disturbed. Logged in `results/steps/steps.txt`.
6. `compare_fixed` takes about 72 minutes per comparison (single-threaded, CPU), 6 of them: most of the 9.9 hours is that, not the HF runs (13 to 25 minutes each).

## Steps, commands, results
Paths are `<src>` the worktree, `<py>` trainer python, `<vpy>` vLLM python, `<ck>` the checkpoint directory.

| Step | Command | Wall | Result |
|---|---|---|---|
| 1 build | `TORCH_CUDA_ARCH_LIST=9.0 <py> setup.py build_ext --inplace`; `<py> -c "import rl_engine._C as c; print(c.__file__)"` | 82 s | rc 0; `c-file.txt` shows the path inside the worktree |
| 2 smoke | `RUN_TP_CUDA=1 <py> -m pytest -rs tests/integrations/common/test_tp_bitwise.py -k nccl` | 41 s | **3 passed**, 4 deselected, 0 skipped |
| 2 smoke | `<py> -m pytest -rs tests/integrations/engines/train/test_qwen3_ppo_checkpoint.py` | 20 s | **22 passed** |
| 3 calibration (1 GPU) | `bash tools/validation/models/qwen3_ppo/qwen3_reference.sbatch <src> <py> <ck> <results>/calibration` | 75 s | rc 0, `thresholds.json`: behavior_logprob atol 0.908, loss atol 0.0389, grad_norm atol 3.034 |
| 4 reference v2 (4 GPUs) | `bash tools/validation/models/qwen3_ppo/qualify_reference.sbatch <src> <py> <vpy> <ck> <thresholds.json> <results>/reference` | 35719 s | rc 0; vLLM TP2 capture, targets, 12 HF runs, 8 bounds files, `acceptance.json` |

Only the `*.pt` files are not in the commit; `rollout/manifest.json`, `targets/targets.json`, all `report.json`, `*-bounds.json`, `acceptance.json`, logs, `slurm.txt`, `device.txt`, `dependencies.txt`, `vllm-dependencies.txt` and `commit.txt` are.

### HF runs (peak allocated GPU memory from each `report.json`, `optimizer_state` is `cpu` in all)
| Run | Peak GiB |
|---|---|
| actor-fp32-0 | 70.4 |
| actor-fp32-1 | 70.4 |
| actor-fp32-impl | 70.7 |
| actor-bf16-0 | 65.7 |
| actor-bf16-1 | 65.7 |
| actor-bf16-impl | 65.7 |
| critic-fp32-0 | 65.7 |
| critic-fp32-1 | 65.7 |
| critic-fp32-impl | 65.7 |
| critic-bf16-0 | 58.8 |
| critic-bf16-1 | 58.8 |
| critic-bf16-impl | 58.8 |

All 12 passed; targets (score only) peak 30.6 GiB. Bounds files: `{actor,critic}-{fp32,bf16}-repeat-bounds.json` and `{actor,critic}-cross-precision-bounds.json` (8 files).

## Step 5: TP2 trainer memory probe (stopped: go rule not met)
Command: `bash tools/validation/models/qwen3_ppo/probe_trainer_memory.sh <src> <py> <overlay> <ck> <reference run dir> <results>/probe 2` (started 2026-11-20T04:17:11Z, rc 0, 4044 s). The probe writes no evidence tensors and no acceptance manifest. Rule: continue only if all 8 rows are `probe_complete` and every rank peak is at most 88 GiB. `probe/probe-*/summary.tsv` and the logs are included.

| Run | Status | Peak GiB per rank | Seconds | Error |
|---|---|---|---|---|
| megatron-fp32-actor-tp2 | probe_complete | 66.1, 66.1 | 675 | |
| megatron-fp32-critic-tp2 | probe_complete | 61.3, 61.3 | 623 | |
| megatron-bf16-actor-tp2 | **failed** | 90.0, 90.0 | 366 | `torch.OutOfMemoryError`: CUDA out of memory, 93.09 GiB capacity, 90.03 GiB allocated by PyTorch (above the 88 GiB limit) |
| megatron-bf16-critic-tp2 | probe_complete | 86.0, 86.0 | 508 | |
| deepspeed-fp32-actor-tp2 | probe_complete | 66.1, 66.1 | 685 | |
| deepspeed-fp32-critic-tp2 | probe_complete | 61.3, 61.3 | 620 | |
| deepspeed-bf16-actor-tp2 | **failed** | 7.6, 7.6 | 128 | `NotImplementedError: unsupported mix of model_dtype=torch.bfloat16 and grad_accum_dtype=torch.float32` (raised in `deepspeed.initialize`, `engine.py` `_do_optimizer_sanity_check`, deepspeed 0.19.7; before any training step) |
| deepspeed-bf16-critic-tp2 | **failed** | 7.6, 7.6 | 128 | the same `NotImplementedError` |

Result: **5 of 8 `probe_complete`; 3 failed.** Two different causes: (1) Megatron BF16 actor at TP2 runs out of memory on a 93 GiB H100 (the 86.0 GiB BF16 critic row fits under 88 GiB, the actor does not); (2) DeepSpeed BF16 does not initialise in this environment (deepspeed 0.19.7, torch 2.9.1) because of the BF16 model with an FP32 gradient accumulation dtype, so those two rows measure no memory. Nothing was rerun or changed.

## Not run
Step 6 (candidate: Megatron/DeepSpeed x FP32/BF16 x actor/critic x TP2/TP4 and the tiny-GPT regression), by the stop rule of step 5. TP4 was never probed (it is outside the probe by design).

## Round 2 (2026-11-22): `19b62661`, from step 5

Done as in `h100-handoff/pr548-item4-20261119/ROUND2.md` (handoff branch at `2027aa21`). Steps 1-4 were not rerun; the frozen reference of round 1 (`results/reference/`, tensors still on disk) is the one used. The round-1 probe directory is unchanged; the new probe wrote `results/probe/probe-20261122T000414Z/`.

### Commit check
- `git -C <src> fetch origin handoff/h100-pr548-item4-20261119` (tip `2027aa21`), `git -C <src> checkout --detach 19b62661`; `git status --porcelain` empty before and after every step (`results/steps/r2-probe.status-after.txt` is empty).
- `<py> -c "import rl_engine._C as c; print(c.__file__)"`: the `_C` built in round 1, inside `<src>` (no `csrc` change between `1bba6d19` and `19b62661`).

### Unit tests (CPU)
`PYTHONPATH=<src>:<overlay> <py> -m pytest -rs -q -p no:cacheprovider tests/integrations/engines/train/deepspeed/test_deepspeed_ppo_zero_stage.py tests/integrations/engines/train/deepspeed/test_deepspeed_bf16_flat_layout.py tests/integrations/engines/train/deepspeed/test_deepspeed_bf16_zero_norm_step.py tests/validation/models/test_qwen3_fixed_matrix.py`: **26 passed** in 27 s (`results/round2/unit-tests.log`).

### Step 5 again: trainer memory probe
`bash tools/validation/models/qwen3_ppo/probe_trainer_memory.sh <src> <py> <overlay> <checkpoint> <reference run dir> <results>/probe` (no TP size argument), on an idle node (4 x H100, 95830 MiB each), 2026-11-22T00:04:13Z to 01:27:28Z, wall 4995 s, rc 0.

**All 8 rows `probe_complete`; highest peak 86.0 GiB per rank (Megatron BF16 critic, TP2), so the go rule (every rank at most 88 GiB) is met.**

| Probe | Status | Peak per rank (H100, round 2) | B200 pre-check | Round 1 on H100 | Seconds |
|---|---|---:|---:|---:|---:|
| Megatron, FP32 actor, TP2 | `probe_complete` | 66.1 GiB | 66.1 GiB | 66.1 GiB | 672 |
| Megatron, FP32 critic, TP2 | `probe_complete` | 61.3 GiB | 61.2 GiB | 61.3 GiB | 597 |
| Megatron, BF16 actor, TP4 | `probe_complete` | 46.6 GiB | 46.6 GiB | (TP2: OOM at 90.0 GiB) | 543 |
| Megatron, BF16 critic, TP2 | `probe_complete` | 86.0 GiB | 86.0 GiB | 86.0 GiB | 529 |
| DeepSpeed, FP32 actor, TP2 | `probe_complete` | 66.1 GiB | 66.1 GiB | 66.1 GiB | 699 |
| DeepSpeed, FP32 critic, TP2 | `probe_complete` | 61.3 GiB | 61.2 GiB | 61.3 GiB | 634 |
| DeepSpeed, BF16 actor, TP4 | `probe_complete` | 53.7 GiB | 53.6 GiB | (TP2: failed at init) | 526 |
| DeepSpeed, BF16 critic, TP4 | `probe_complete` | 49.6 GiB | 49.6 GiB | (TP2: failed at init) | 453 |

Every rank of a row has the same peak (`summary.tsv`); H100 and the B200 pre-check agree within 0.1 GiB on all 8 rows.

### Step 6: candidate (stopped at run 2: one tensor over its bound)
The probe above was committed and pushed first, as ROUND2.md asks (`fb9a4538`).

`bash tools/validation/models/qwen3_ppo/qualify_candidate.sbatch <src> <trainer-python> <vllm-python> <overlay> <checkpoint> <reference run dir> <reference run dir>/acceptance.json <results>/candidate`, unchanged, on an idle node (4 x H100, no other process; `results/steps/candidate.nvidia-smi-start.txt`), 2026-11-22T03:44:30Z to 06:18:58Z, wall 9269 s, **rc 1**. The tree stayed clean (`results/steps/candidate.status-after.txt` is empty). Run directory: `results/candidate/job-JOBID-attempt-0/` (tensors stay on disk).

`verify_reference` on the frozen reference passed (about 52 min, CPU). Then:

| # | Run | Status | Peak allocated per rank | Run seconds | Against frozen FP32 repeat bounds |
|---|---|---|---:|---:|---|
| 1 | Megatron, FP32 actor, TP2 | `fixed_candidate_complete` | 66.1 GiB | 1322 | **passed** (1604 tensors; tightest 0.82 of its bound, `0/update/model.layers.8.input_layernorm.weight`) |
| 2 | Megatron, FP32 actor, TP4 | `fixed_candidate_complete` | 35.6 GiB | 1209 | **failed**: 1 of 1604 tensors over its bound |

The one failure (`megatron-fp32-actor-tp4/comparison.json`, `failures`):

| Tensor | max_abs (TP4) | bound (`actor-fp32-repeat-bounds.json`) | ratio | same tensor at TP2 |
|---|---:|---:|---:|---:|
| `0/update/model.layers.17.post_attention_layernorm.weight` | 7.749e-07 | 3.384e-07 | 2.29 | 2.384e-07 (0.70) |

Every other TP4 tensor stays at or below 0.53 of its bound (next: `0/update/model.layers.24.post_attention_layernorm.weight`, 0.53). The relative L2 of the failing tensor is 1.22e-03 at TP4 against 3.94e-04 at TP2. The two runs' losses (-0.0712526 vs -0.0712537) and gradient norms (76.9235 vs 76.9238) are close.

As ROUND2.md says, the script stops at the first failing comparison; this is the result, kept as it is and not rerun. The script runs with `set -euo pipefail`, so it exited right after run 2's comparison. Not reached: the Megatron FP32 actor TP2-vs-TP4 comparison (next in the loop), the Megatron FP32 critic, both Megatron BF16 rows, all DeepSpeed rows, the other four TP2-vs-TP4 comparisons, and the tiny-GPT regressions.

Directory names: the job-local run directories are written `job-JOBID-attempt-N`.
