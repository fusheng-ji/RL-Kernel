# H100 handoff: PR #536 (HAL persistent autotuning) at its current head (2026-11-19)

## What and why

PR #536 (issue #50) adds persistent kernel autotuning to the HAL. Its head moved from
`4a83e4f8` (measured on H100 on 2026-11-18) to **`2ef6e6ff`**. That commit answers Flink-ddd's
2026-11-19 review: if a profiling run raises `torch.cuda.OutOfMemoryError`,
`TunableKernel.launch` now falls back to the fixed launch and caches nothing. It also adds
regression tests for that. Changed files since `4a83e4f8`:
`rl_engine/backends/shared/triton/autotuning.py`, `rl_engine/runtime/registry.py`,
`tests/platforms/test_persistent_autotuner.py`, `docs/usage/autotuning.md`.

This run needs to give us:

1. **The autotune suite at `2ef6e6ff`**, including the new tests
   `test_launch_falls_back_when_profiling_runs_out_of_memory` and
   `test_launch_propagates_profiling_errors_that_are_not_out_of_memory`.
2. **`tests/ops` compared with `upstream/main`, one test file per process.** On 2026-11-18 both the
   head and main hit a CUDA illegal memory access in
   `tests/ops/gemm/test_det_gemm.py::test_target_shapes_invariance[shape3-cuda-deterministic_gemm]`.
   That error is sticky: every later CUDA test in the same process failed too, so the counts
   (816 failed on the head, 332 on main) only measured how far the crash spread. One file per process
   keeps a crash inside its own file, so the head and main can be compared file by file.
3. **That one crashing test alone,** on head and main, with `CUDA_LAUNCH_BLOCKING=1`, plus one try
   with the SM90 GEMM build flags. This tells us whether the crash comes from the build or from
   `main`. We do not expect it to come from this PR.

## What to check out

- Head: fork `https://github.com/fusheng-ji/RL-Kernel`, branch `feat/issue-50-hal-autotuning` at
  **`2ef6e6ff`**. Before you start, run `gh pr view 536 -R RL-Align/RL-Kernel --json headRefOid`. If the
  head is no longer `2ef6e6ff`, stop and ask.
- Before: `upstream/main` (`https://github.com/RL-Align/RL-Kernel`) at **`49941801`**, the merge base
  of the head. Record the SHA you use. If `main` has moved, still use `49941801`.
- Two detached worktrees, `<head>` and `<main>`: `git worktree add --detach <dir> <sha>`. Do not edit
  either one. `git status --porcelain --untracked-files=no` must be empty before and after every
  step. Build output is ignored by git.

## Hardware and environment

- One H100 node with 4 GPUs, nothing else running on it (TP=4 cases need all 4 GPUs visible). Use
  the same environment as the 2026-11-18 evidence-gaps run: python 3.12, torch 2.13.0+cu130,
  triton 3.7.1, `ninja` on `PATH`, `PYTHONNOUSERSITE=1`. Do not install anything into the shared
  environment.
- Header at the top of every result: GPU name and count, driver, CUDA, torch, triton, python,
  the SHA of the worktree, `git status`, exact commands, and whether another job was on the node.
- On 2026-11-18, `test_det_gemm.py` on `main` reported 40 skips as "Triton is unavailable", while
  the head had none. In each worktree, record
  `python -c "import triton, rl_engine; print(triton.__version__, triton.__file__)"` (paths redacted)
  and say whether those skips appear again.

## Steps

Run every command from inside the worktree it is about, with `PYTHONPATH=.`. Use a new private
autotune cache directory for every command
(`RL_KERNEL_AUTOTUNE_CACHE_DIR=$(mktemp -d)`) and `RL_KERNEL_MODE=auto`.

1. **Build both worktrees** the same way: `TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace`.
   Before each GPU step, `python -c "import rl_engine._C as c; print(c.__file__)"` must print a path
   inside the worktree of that step. Save it as `c-file.txt`.

2. **Autotune suite on the head (about 12 minutes).**
   ```bash
   python -m pytest -p no:cacheprovider -q -rs tests/platforms/test_persistent_autotuner.py \
     tests/runtime/test_kernel_registry.py tests/ops/test_autotuning_gpu.py \
     tests/ops/test_autotuning_candidates.py tests/benchmarks/test_autotuning_benchmark.py
   ```
   Also run `python -m pytest -p no:cacheprovider -v -rs tests/platforms/test_persistent_autotuner.py -k out_of_memory`
   and keep the `-v` log, so each of the two new tests appears by name with its outcome.

3. **`tests/ops` one file per process, head and main, `RL_KERNEL_AUTOTUNE=0` and `=1` (4 runs).**
   ```bash
   for f in $(find tests/ops -name 'test_*.py' | sort); do
     RL_KERNEL_AUTOTUNE=$A RL_KERNEL_AUTOTUNE_CACHE_DIR=$(mktemp -d) \
       python -m pytest -p no:cacheprovider -q -rs "$f" > "$OUT/ops-autotune$A/$(echo "$f" | tr / _).log" 2>&1
     echo "$f rc=$?" >> "$OUT/ops-autotune$A/rc.txt"
   done
   ```
   Hand back a table with one row per file: passed / failed / skipped / error for head and main,
   `AUTOTUNE=0` and `=1`. Mark each row as **same**, **head only** (fails on the head, passes or is
   missing on main) or **main only**. Files that exist on only one side are listed separately.
   **Every "head only" failure: copy its first traceback into the summary.** That is what we need to
   know.

4. **The crashing test alone, head and main (a few minutes).**
   ```bash
   CUDA_LAUNCH_BLOCKING=1 python -m pytest -p no:cacheprovider -v -rs \
     "tests/ops/gemm/test_det_gemm.py::test_target_shapes_invariance"
   ```
   Keep the full log, including the first traceback. Then **once**, in a third detached worktree at
   `2ef6e6ff`, build with
   `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace`
   and run the same command, plus the whole `tests/ops/gemm/test_det_gemm.py`. Report it as a
   separate build, not as the head's result.

Stop rule: a failure is a result. Keep it as it is, do not retry until green, do not change any
file in a worktree.

## Returning results

- Push to the fork branch `handoff/h100-pr536-results-20261119` only. Directories:
  `results/pr536/2ef6e6ff/`, `results/main-49941801/` and `results/pr536/2ef6e6ff-sm90-gemm/`. Do not
  push to the PR branch or `pr_assets`. Do not open a PR or comment anywhere.
- Start with `results/SUMMARY.md`:
  - the header;
  - the step 2 result line and the two new tests by name;
  - the step 3 per-file table, with the "head only" tracebacks;
  - the step 4 results for the three builds;
  - a drafted "H100" paragraph for the PR description that names `2ef6e6ff` and states each
    comparison against `upstream/main`.
- Include logs, `rc.txt`, `c-file.txt` and the headers. `*.log` is in `.gitignore`, so use `git add -f`.
- Redact before committing: cluster, partition, job and node names, hostnames, absolute paths
  (use `<HOME>` / `<WORK>`), usernames, emails, IPs, and GPU UUIDs and PCI bus ids. Autotune cache
  dumps contain the device UUID and PCI bus id.
- Commits: `git commit -s`, no AI attribution lines.
