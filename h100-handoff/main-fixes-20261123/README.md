# H100 handoff: Hopper results for the five `main` fixes (#746 – #750) and the vLLM sort top-k/top-p fix (2026-11-23)

## Why

Six small `main` fixes were measured on B200 only. Their PR descriptions need a "Results (H100 / Hopper)" section:

| Tree | What it fixes | B200 result (reference/b200/) |
|---|---|---|
| `main` `e9d52e95` | the baseline | 72 failed / 3835 passed / 974 skipped (with the AITER contract file excluded) |
| #746 `921f6853` | AITER `mha_fwd` `out=` check; the contract test file imports again | same 72 failures as `main`, +3 passed +3 skipped (the file runs) |
| #747 `1f93314b` | `get_op("rms_norm")` returns the CUDA op | −1 failure (`test_registry_dispatches_rms_norm`); plus a speed / bit comparison (step D) |
| #748 `219ad592` | explicit attention / logprob requests no longer report `fallback=True` | −2 failures (`test_attention_dispatch.py`) |
| #749 `acae3a40` | NVRTC found in CUDA 13 wheels | −7 failures (`test_reproducible_norm.py`) |
| #750 `de996160` | stale ROCm FFN test | `tests/layers/`: 2 failed → 0 |
| G `95518dbe` (no PR yet) | vLLM samples top-k/top-p with the sort that strict scoring reproduces | −3 failures (capture test), +8 new tests; plus a support / cost comparison (step E) |

On B200, 51 of `main`'s 72 failures are `cublaslt_nosplitk is currently validated only on SM90`. On H100 those should run, so the H100 baseline list will be different. What matters is the **difference** between each tree and the H100 baseline.

Rules 1-6 of `h100-handoff/evidence-gaps-20261114/README.md` (fork branch `handoff/h100-evidence-gaps-20261114`, "Rules for every run") apply. In short:
- one clean, detached worktree per tree; never edit tracked files;
- check `rl_engine._C.__file__` before every GPU run;
- failures are results: don't retry until green;
- redact everything before it goes back.

Results go to a results branch (below), not to `pr_assets`.

## Code

Five trees come from the fork, `main` comes from upstream, and G comes from the bundle in this directory. The bundle holds `refs/heads/fix/vllm-sort-topk-topp-contract` at `95518dbe`; its prerequisite is `e9d52e95` (upstream `main`).

```bash
cd <REPO>
git fetch https://github.com/RL-Align/RL-Kernel.git main                       # provides e9d52e95
git fetch https://github.com/fusheng-ji/RL-Kernel.git \
  fix/rocm-aiter-fwd-out-contract fix/registry-cuda-rmsnorm-main \
  fix/attention-dispatch-explicit-id-fallback fix/nvrtc-cuda13-lookup fix/rocm-ffn-test-module
git bundle verify <THIS_DIR>/vllm-sort-topk-topp-20261123.bundle
git fetch <THIS_DIR>/vllm-sort-topk-topp-20261123.bundle \
  'refs/heads/fix/vllm-sort-topk-topp-contract:refs/heads/h100/vllm-sort-topk-topp'
for t in main:e9d52e95 pr746:921f6853 pr747:1f93314b pr748:219ad592 pr749:acae3a40 pr750:de996160 sort:95518dbe; do
  git worktree add --detach <WT>/${t%%:*} ${t#*:}
  (cd <WT>/${t%%:*} && git config core.fileMode false && git status --porcelain --untracked-files=no)   # must be empty
done
```

Re-check the five PR heads with `gh pr view N --repo RL-Align/RL-Kernel --json headRefOid` before starting. If one moved, test the SHA the PR shows and say so.

## Environment

Use the RL-Kernel environment of the previous rounds (torch 2.13.0+cu130, vLLM 0.30.0) unchanged, and install nothing.
- Use `PYTHONNOUSERSITE=1` for every step.
- Record `pip list 2>/dev/null | grep -iE "^(torch|triton|vllm|flashinfer)"` in `<OUT>/env.txt`.

**Step E depends on the vLLM version:**
- vLLM 0.30 uses the Triton top-k/top-p kernel for every row count;
- vLLM 0.17 – 0.27 use it from 8 rows;
- vLLM 0.16 has no Triton path.

Run every step with `python -m pytest` or plain `python` from inside the tree, with `PYTHONPATH=<that tree>`. Do not use `python -I`: it ignores `PYTHONPATH` and imports the editable install.

## A. Build once, share the extension (about 10 min, 1 GPU)

None of the seven trees changes `csrc/` or `setup.py` relative to `e9d52e95`, so one build serves all of them.

```bash
cd <WT>/main
for t in pr746 pr747 pr748 pr749 pr750 sort; do git -C <WT>/$t diff --quiet e9d52e95 HEAD -- csrc setup.py || echo "CSRC DIFFERS: $t"; done   # prints nothing
TORCH_CUDA_ARCH_LIST=9.0 KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 MAX_JOBS=32 \
  python setup.py build_ext --inplace > <OUT>/A/build.log 2>&1
for t in pr746 pr747 pr748 pr749 pr750 sort; do cp rl_engine/_C.cpython-*.so <WT>/$t/rl_engine/; done
for t in main pr746 pr747 pr748 pr749 pr750 sort; do
  (cd <WT>/$t && PYTHONPATH=$PWD python -c "import rl_engine._C as c; print('$t', c.__file__)"; git status --porcelain --untracked-files=no)
done > <OUT>/A/c-file.txt 2>&1                                                  # every path under its own tree; no status lines
```

Use the same build flags as the earlier H100 rounds. If the flags differ, record them.

## B. Full suite on every tree (about 10 min per tree, 1 GPU)

```bash
for t in main pr747 pr748 pr749 pr750 sort; do
  cd <WT>/$t
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m pytest tests/ --ignore=tests/distributed \
    --ignore=tests/backends/rocm/test_rocm_aiter_api_contract.py -q --tb=no -rf -p no:cacheprovider > <OUT>/B/$t.log 2>&1
done
cd <WT>/pr746                                     # #746 runs the AITER contract file too
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m pytest tests/ --ignore=tests/distributed -q --tb=no -rf -p no:cacheprovider > <OUT>/B/pr746.log 2>&1
cd <WT>/main                                      # main with the AITER file: should stop with 1 collection error
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m pytest tests/ --ignore=tests/distributed -q --tb=no -p no:cacheprovider > <OUT>/B/main-with-aiter-file.log 2>&1
for t in main pr746 pr747 pr748 pr749 pr750 sort; do grep '^FAILED' <OUT>/B/$t.log | sed 's/ - .*//; s/^FAILED //' | sort > <OUT>/B/$t.fails; done
for t in pr746 pr747 pr748 pr749 pr750 sort; do echo "== $t fixed:"; comm -23 <OUT>/B/main.fails <OUT>/B/$t.fails; echo "== $t new:"; comm -13 <OUT>/B/main.fails <OUT>/B/$t.fails; done > <OUT>/B/diff.txt
```

Do not pipe pytest through `tail`: the `-rf` summary is longer than it looks.

**Expected in `diff.txt`:**
- **pr746:** nothing fixed, nothing new.
- **pr747:** fixed `tests/ops/norm/test_rms_norm.py::test_registry_dispatches_rms_norm`.
- **pr748:** fixed the two `tests/runtime/test_attention_dispatch.py` tests.
- **pr749:** fixed the 7 `tests/ops/norm/test_reproducible_norm.py` accelerator tests.
- **pr750:** fixed the 2 `tests/layers/test_canonical_tp_ffn.py::test_rocm_canonical_ffn_resolves_aot_slot_before_ipc_reduction` cases.
- **sort:** fixed the 3 `test_captured_full_support_matches_independent_policy[1|4|64]` ids (renamed; the 5 new ids pass).

Every tree should show **no new failures**. If a test fails differently on H100 than on B200 (for example because the SM90 GEMM path runs), list it; that is a finding. The B200 lists are in `reference/b200/failures-*.txt`.

## C. Targeted checks with the fix reverted (about 10 min)

This confirms on Hopper that each new test catches its bug. For each tree, put the library file from `e9d52e95` into a **plain copy** of the tree, then run the targeted tests. Use `git show` to write the old file: `git checkout` in a copied worktree would change the original worktree's index, because they share a git directory.

| Tree | Revert | Tests | B200: with fix / reverted |
|---|---|---|---|
| pr747 | `rl_engine/runtime/registry.py` | `tests/runtime/test_kernel_registry.py tests/ops/norm/test_rms_norm.py` | 68 passed / 4 failed |
| pr748 | `rl_engine/runtime/registry.py` | `tests/runtime/test_attention_dispatch.py tests/contracts/operators/test_logprob_contract.py` | 71 passed / 7 failed |
| pr749 | `rl_engine/ops/norm/reproducible.py` | `tests/ops/norm/test_reproducible_norm.py` | 17 passed / 10 failed |
| sort | `rl_engine/integrations/engines/rollout/vllm/sampling_support.py` | `tests/backends/cuda/test_decode_optimizations.py -k captured_full_support` | 5 passed / 5 failed |

```bash
cp -a <WT>/<t> <SCRATCH>/<t>-reverted && cd <SCRATCH>/<t>-reverted && git -C <WT>/<t> show e9d52e95:<file> > <file>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m pytest <tests> -q -p no:cacheprovider --tb=no > <OUT>/C/<t>-reverted.log 2>&1
```

## D. #747: native vs CUDA `rms_norm` (about 10 min, 1 idle H100)

Timing needs a GPU with no other job on it. Say in `SUMMARY.md` whether that was the case.

```bash
cd <WT>/pr747
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python <THIS_DIR>/scripts/rmsnorm_dispatch_bench.py <OUT>/D/rmsnorm-dispatch-h100.json <WT>/main > <OUT>/D/bench.log 2>&1
```

**Expected:**
- `registry_choice` is `{"branch": "RMSNormCudaOp", "main": "NativeRMSNormOp"}`.
- `*_run_to_run_bitwise` and `*_batch_invariant` are true for both ops at every shape.

**Reference (B200, `reference/b200/rmsnorm-dispatch-b200.json`):**
- the CUDA forward is 3.3–8.4x faster;
- CUDA forward+backward is faster up to 256 tokens and 2.0–3.4x slower from 4096 tokens;
- native and CUDA outputs are bitwise equal at 1 and 16 tokens, and differ from 256 tokens up (max 1.6e-2 – 3.1e-2), with the same FP64 error.

Whatever the H100 ratios are, they are the result.

## E. vLLM Triton top-p support vs the sort (about 10 min, 1 idle H100)

```bash
cd <WT>/sort
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python <THIS_DIR>/scripts/topp_contract_bench.py <OUT>/E/topp-contract-h100.json > <OUT>/E/bench.log 2>&1
```

**Expected:**
- `sort_equals_policy` true for every `top_p`;
- `rows_differing_triton` above 0 on vLLM 0.30.

**Reference (B200, `reference/b200/topp-contract-b200.json`):**
- 53–60 of 64 rows differ, with up to 10–38 extra tokens per row;
- Triton is a superset of the sort up to p = 0.95, but not at 0.99;
- batch dependence: Triton 23 of 128 cases, sort 2 of 128;
- cost per call at 1 / 8 / 64 / 256 rows, Triton vs sort: 0.11 / 0.12 / 0.61 / 1.54 ms vs 0.16 / 0.44 / 1.04 / 3.50 ms.

## F. Optional: the 8-GPU sampling check (only if an idle 8 x H100 node is available, about 10 min)

The G tree changed `tests/integrations/engines/rollout/vllm/distributed/check_configurable_sampling.py` to install the sort contract. It has not run on 8 GPUs anywhere yet.

```bash
cd <WT>/sort
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=8 \
  tests/integrations/engines/rollout/vllm/distributed/check_configurable_sampling.py > <OUT>/F/sort.log 2>&1; echo "rc=$?" >> <OUT>/F/sort.log
cd <WT>/main                                      # the same script at main, for comparison
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=8 \
  tests/integrations/engines/rollout/vllm/distributed/check_configurable_sampling.py > <OUT>/F/main.log 2>&1; echo "rc=$?" >> <OUT>/F/main.log
```

**Expected:**
- on the G tree, `rc=0`;
- at `main` on vLLM 0.30, the assertion `support differs from vLLM` fails (on B200, 2 of its 4 configurations differ on one GPU).

Skip this step if no such node is free, and say so.

## Send back

Push to a new fork branch `handoff/h100-main-fixes-results-20261123`, under `results/<A..F>/`. Use `git add -f` for `*.log`.

Include:
- `env.txt` and `A/c-file.txt`;
- `nvidia-smi --query-gpu=name,driver_version --format=csv`;
- for B: every `*.log`, `*.fails` and `diff.txt`;
- the C logs;
- the D and E JSON and logs;
- F, if it ran;
- a `SUMMARY.md` with one row per step and per tree: passed / failed / skipped, fixed, new.

Keep failed attempts under `not-green-attempts/`, unedited.

Redact host names, user names, absolute paths, job ids, cluster and environment names, GPU UUIDs and PCI bus ids before pushing. Do not open a PR, edit a PR description or post comments.
