# PR #548 review item 4: Qwen3-8B fixed-rollout qualification on H100 (2026-11-19/20)

**Status of this commit: steps 1-4 only (build, smoke, calibration, reference v2). No probe and no candidate has run yet.** This commit contains everything in the reference directory except tensor files. It exists to show that `acceptance.json` and all `*-bounds.json` were written (2026-11-20 04:41 CET at the latest) before any probe or candidate run started. The tensors (about 1.4 TB) stay on disk. `SUMMARY.md` is extended by later commits for steps 5 and 6.

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
