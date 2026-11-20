# H100 handoff: PR #548 review item 4, Qwen3-8B fixed-rollout qualification (2026-11-19)

Review item 4 (Flink-ddd, 2026-11-02): "complete the fixed-rollout Qwen3 qualification after
fixing the reductions and gradient synchronization." Items 1-3 are fixed in `cb2fd250`,
`194b82eb` and `548706f3`. This run produces item 4 on one node with four H100 (80-94 GB).

## What to check out

- Fork `https://github.com/fusheng-ji/RL-Kernel`, this branch
  `handoff/h100-pr548-item4-20261119`. The run code is commit **`b9e5690d`** (updated
  2026-11-20); the rest of the branch only adds this directory.
- Make a detached worktree at `b9e5690d` and never edit it:
  `git worktree add --detach <src> b9e5690d`. Every script checks that `git status --porcelain`
  is empty. Build output (`rl_engine/_C*.so`) is ignored by git.
- **If steps 1-4 already ran at `1bba6d19`, keep them and do not rerun them.** Before step 5,
  run `git -C <src> checkout --detach b9e5690d`. `b9e5690d` changes only the probe and
  candidate scripts, a document and a test: no reference code and no `csrc`, so the built
  `_C` and the frozen reference stay valid. Record in `SUMMARY.md` which commit ran each step.
- `b9e5690d` is `1bba6d19` plus one change: the Megatron BF16 actor row runs at TP4 only. Its
  TP2 probe peaked at 92.9 GiB per rank on B200, above the 88 GiB limit of step 5
  (`B200-CHECK.md`).
- `1bba6d19` sits on `548706f3` (items 1-3) + `9e64f583` (implementation-spread control) +
  `df61088f` (control recomputes its linear layers in backward) and makes the run fit a
  94 GB GPU:
  - the HF reference keeps its AdamW moments and BF16 FP32 masters in host memory between
    steps and applies the same per-tensor `torch.optim.AdamW` update on the GPU, one
    parameter at a time (values unchanged: a CUDA unit test checks this bit for bit, see
    "What was checked before");
  - the candidate matrix is TP2/TP4 (TP1 needs about 131 GB), the Megatron BF16 actor TP4
    only (`tools/validation/models/qwen3_ppo/fixed_matrix.sh`);
  - `check_qwen3_fixed --probe` and `probe_trainer_memory.sh` measure trainer memory before
    any candidate run.

## What was checked before (B200)

- Unit tests on B200: the host-resident optimizer equals one resident AdamW bit for bit over
  two steps with two microbatches, FP32 and BF16, state on CPU and on GPU (breaking the state
  load makes it fail); the TP2-vs-TP4 comparison accepts only the frozen sizes in increasing
  order; `tests/integrations/common` + `tests/integrations/engines/train`: 210 passed,
  5 skipped; mypy clean.
- Full Qwen3-8B check of `1bba6d19` on B200 (2026-11-20, `B200-CHECK.md`): optimizer state on
  the CPU equals state on the GPU bit for bit (actor FP32 and BF16, 1604 tensors each); with
  the state on the CPU every HF run peaks at 70.3 GiB or less; probes: Megatron BF16 actor TP2
  92.9 GiB per rank (hence `b9e5690d`), DeepSpeed FP32 critic TP4 33.0 GiB. The other rows
  were not probed on B200.
- `b9e5690d`: the matrix test and the dry runs of both scripts with a python stand-in
  (`tests/validation/models/test_qwen3_fixed_matrix.py`, no GPU) pass, 12 tests.
- Earlier B200 HF peaks with the optimizer on the GPU: 157 GiB (actor BF16 implementation
  control) and 143 GiB (critic BF16 control); about 11-12 minutes per HF run.

## Hardware and environment

- 4 x H100 on one node, nothing else running on it during steps 3-6. Report the GPU model,
  memory per GPU, driver and CUDA.
- Host: at least 256 GB free RAM (the BF16 reference holds about 130 GB of optimizer state and
  masters in RAM); disk: about 1.6 TB free for the reference.
- Three environments, the same versions as the earlier B200 runs (`setup_envs.sh` in this
  directory shows how they were built; use a CUDA build of torch that supports SM90):
  - trainer: Python 3.11, torch 2.9.1, transformers 5.13.1, plus pytest and ninja;
  - framework overlay: a `pip install --no-deps --target <overlay>` directory with
    `examples/post_training/tp_ppo/requirements-frameworks.txt` (deepspeed 0.19.7,
    megatron-core 0.15.3), used only through `PYTHONPATH`;
  - vLLM: a separate Python 3.11 environment with vllm 0.15.1.
- Model: `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`. Every
  script checks all files against `tools/validation/models/qwen3_ppo/qwen3-8b-manifest.json`.
- Set for every step:
  - `export PYTHONNOUSERSITE=1` (a user-site package shadowed the environment on the H100
    side before);
  - `export TMPDIR=/tmp/q548-$$; mkdir -p $TMPDIR` (vLLM's IPC paths fail with a long TMPDIR);
  - `ninja` on `PATH` (vLLM's FlashInfer all-reduce compiles at start).

## Steps (stop rules in bold)

1. **Build `_C` in the worktree.**
   `cd <src> && TORCH_CUDA_ARCH_LIST=9.0 <trainer-python> setup.py build_ext --inplace`, then
   `<trainer-python> -c "import rl_engine._C as c; print(c.__file__)"` must print a path
   inside `<src>`. Save the output as `c-file.txt`.

2. **Smoke (about 5 minutes).** From `<src>` with `PYTHONPATH=<src>`:
   - `RUN_TP_CUDA=1 <trainer-python> -m pytest -rs tests/integrations/common/test_tp_bitwise.py -k nccl`
   - `<trainer-python> -m pytest -rs tests/integrations/engines/train/test_qwen3_ppo_checkpoint.py`
     (includes the CUDA test that the host-resident optimizer equals a resident AdamW bit for bit).

   **Any failure: stop and hand back.**

3. **Calibration (1 GPU, under 3 hours).**
   `bash tools/validation/models/qwen3_ppo/qwen3_reference.sbatch <src> <trainer-python> <checkpoint> <results>/calibration`
   It writes `<results>/calibration/<run>/reference/thresholds.json`.

4. **Reference v2 (vLLM TP2 capture, then 12 HF runs on 1 GPU each).**
   `bash tools/validation/models/qwen3_ppo/qualify_reference.sbatch <src> <trainer-python> <vllm-python> <checkpoint> <thresholds.json from step 3> <results>/reference`
   - It captures a fresh vLLM rollout, computes HF FP32 targets, runs actor/critic x FP32/BF16
     x {repeat 0, repeat 1, implementation control}, writes the bounds and freezes
     `acceptance.json`. About 1.4 TB.
   - Each `report.json` records `peak_allocated_bytes` and `optimizer_state: "cpu"`.
   - **If any run fails (for example out of memory), the script stops. Stop and hand back;
     do not rerun with other settings.**
   - **Then, before step 5:** commit to the results branch everything in the reference
     directory except tensor files: every `report.json`, the `*-bounds.json` files,
     `acceptance.json`, `rollout/manifest.json`, `targets/targets.json`, the logs and the
     environment files, and push it. This commit is the proof that the bounds were frozen
     before any candidate ran. Keep the tensors on disk until we confirm.

5. **Trainer memory probe (8 runs).**
   `bash tools/validation/models/qwen3_ppo/probe_trainer_memory.sh <src> <trainer-python> <overlay> <checkpoint> <reference run dir> <results>/probe`
   - No TP size argument: it probes each engine/precision/role once at its smallest TP size,
     seven rows at TP2 and the Megatron BF16 actor at TP4, with `--probe`: no evidence tensors, no
     acceptance manifest, so it cannot touch a bound. It keeps going past a failure and
     writes `summary.tsv` (status and peak GiB per rank).
   - **Go on to step 6 only if all 8 rows are `probe_complete` and every rank peak is at
     most 88 GiB. Otherwise stop and hand back with `summary.tsv` and the logs.** The seven
     TP2 rows are not probed at TP4; it needs about half of TP2.

6. **Candidate.**
   `bash tools/validation/models/qwen3_ppo/qualify_candidate.sbatch <src> <trainer-python> <vllm-python> <overlay> <checkpoint> <reference run dir> <reference run dir>/acceptance.json <results>/candidate`
   - Megatron and DeepSpeed x FP32/BF16 x actor/critic x TP2/TP4 against the frozen bounds,
     except the Megatron BF16 actor at TP4 only (15 runs); a TP2-vs-TP4 comparison for each of
     the other 7 rows; then the tiny-GPT regression (4 GPUs).
   - The script stops at the first failing comparison (exit 1). A failure is a result: keep it
     as it is, do not rerun it, and hand back.
   - The pretrained two-round vLLM topologies are not run (they do not fit 4 x 94 GB).

## Returning results

- Branch `handoff/h100-pr548-item4-results-20261119` on the fork, directory `results/`
  (`calibration/`, `reference/`, `probe/`, `candidate/`). Push only that branch: not the PR
  branch, not `pr_assets`, no PR, no comment.
- Include JSON, logs, `summary.tsv`, `steps`/`header` files and `c-file.txt`; leave tensor
  files (`*.pt`, `*.safetensors`, tensor directories) on disk until we confirm. `*.log` is
  in `.gitignore`: use `git add -f`.
- Start with a `SUMMARY.md`: GPU model and count, memory per GPU, host RAM, driver, CUDA,
  torch/vLLM/Megatron/DeepSpeed versions, the commit check (`b9e5690d`, and `1bba6d19` for
  steps that ran there), the full command of each step, wall time per step, and pass/fail per
  step with the peak memory of each HF run and each probe row.
- Redact before committing: cluster, partition, job and node names, hostnames, absolute
  paths, usernames, emails, IPs, GPU UUIDs and PCI bus ids, including inside JSON `command`
  fields, `slurm.txt`, `device.txt` (`nvidia-smi -q`) and `dependencies.txt`.
- Commits: `git commit -s`, no AI attribution lines.
