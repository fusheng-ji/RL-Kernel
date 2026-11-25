# H100 handoff: `h3_qk_rmsnorm_d128` on Hopper (2026-11-25)

## Why

RFC #420's QK RMSNorm row has been measured on B200 only. On H100 this checks that the row's forward is still the provider's bytes, and repeats the survey that backs the reuse decision. The row adds no native code: it uses the merged `h3_rmsnorm` kernel.

| Task | What | Hardware | Time (est.) |
|---|---|---|---|
| **Q1** | tests: the row, the block and refiner graphs, the TP path | 1 H100, idle | 20 min |
| **Q2** | survey: prior art, and the byte comparison | the same card | 30 min |

Paths are placeholders, as in the earlier H3 rounds:
- `<REPO>`: a clone of the fork; `<THIS_DIR>`: this folder; `<W>`: the worktree below; `<OUT>`: where results are written;
- `<E>`: the environment of the earlier H3 rounds (torch 2.13.0+cu130 or 2.14.0+cu130, triton; say which);
- `<WEIGHTS>`: the directory that holds `hf_full/`, as in the earlier rounds;
- `<PROVIDER>`: the Diffusers checkout at `f53d552036a0d1bd5570782a39cd40cfabf112bc`, clean.

## Rules (unchanged)

- **Clean tree.** Detached worktree; `git status --porcelain --untracked-files=no` empty before and after. Change no file.
- **Check `_C`.** `rl_engine._C.__file__` must be inside `<W>`.
- **Idle hardware.** One idle H100; record `nvidia-smi` at the start of each task.
- **Report failures as they are**, with logs, in `not-green-attempts/`. Change no test or bound. Install nothing into a shared environment.
- **Push only the results branch.** No PR, no comment. Redact cluster, job, node, user, environment name, paths and email (hosts become `node-a`). `git add -f` the logs. Commit with `git commit -s`, no AI attribution lines.

## Code

```bash
cd <REPO>
git fetch https://github.com/RL-Align/RL-Kernel.git test-h3
git fetch https://github.com/fusheng-ji/RL-Kernel.git handoff/h100-h3-qk-rmsnorm-20261125
git bundle verify <THIS_DIR>/h3-qk-rmsnorm-20261125.bundle
git fetch <THIS_DIR>/h3-qk-rmsnorm-20261125.bundle 'refs/heads/feat/h3-qk-rmsnorm-d128:refs/h100/qk'
git worktree add --detach <W> refs/h100/qk      # c56b81c4f61ad725c361c964e466d6c256e878a5
cd <W> && RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 KERNEL_ALIGN_DET_GEMM_SM90=1 <E>/bin/python setup.py build_ext --inplace
<E>/bin/python -c "from rl_engine.backends.extension import _C; print(_C.__file__)" > <OUT>/c-file.txt
```

The bundle's prerequisite is `test-h3` `13d0f906`. If the build flag set above differs from your earlier H3 builds, use yours and say so.

## Q1. Tests

```bash
cd <W>; export PYTHONPATH=<W>:<PROVIDER>/src RL_KERNEL_H3_WEIGHTS=<WEIGHTS>
CUDA_VISIBLE_DEVICES=0 <E>/bin/python -m pytest -v -p no:cacheprovider -rfs \
  tests/models/minimax_h3/test_h3_qk_rmsnorm.py tests/models/minimax_h3/test_h3_rmsnorm.py \
  tests/models/minimax_h3/test_h3_one_block.py tests/models/minimax_h3/test_h3_token_refiner.py \
  tests/models/minimax_h3/test_h3_operator_trace.py tests/models/minimax_h3/test_h3_parallel.py \
  tests/models/minimax_h3/test_h3_parallel_model.py tests/models/minimax_h3/test_h3_full_model.py \
  tests/runtime/test_dispatch.py tests/validation/operators > <OUT>/Q1/tests.txt 2>&1
```

**Expected:**
- every test in `test_h3_qk_rmsnorm.py` passes, including the bitwise comparisons with `F.rms_norm` in BF16, FP16 and FP32;
- no failure other than, possibly, the refiner's SwiGLU tolerance case `test_nodes_accuracy_repeat_batch_and_isolated[1-257]` (it fails on B200, where it also fails at the base) and the plot test if matplotlib is missing. Any other failure: run the same test at `13d0f906` in a second worktree and report both results.

B200 reference: 554 passed, 3 failed, 32 skipped at `944a585d`. The same three fail at the base `13d0f906` on B200: the SwiGLU case, the plot test (no matplotlib there), and `test_ws1_candidate_evidence.py::test_ws1_cuda_and_triton_candidate_runtime_provenance`, because that B200 build lacked `KERNEL_ALIGN_DET_GEMM_SM90=1`. The build line above sets it.

## Q2. Survey

```bash
cd <W>; export PYTHONPATH=<W>:<PROVIDER>/src
CUDA_VISIBLE_DEVICES=0 <E>/bin/python tools/validation/models/h3_prior_art.py --op qk_norm --out <OUT>/Q2/qk_norm.json > <OUT>/Q2/prior-art.log 2>&1
CUDA_VISIBLE_DEVICES=0 <E>/bin/python tools/validation/models/h3_bitwise_reuse.py --op qk_norm --out <OUT>/Q2/qk_norm_bitwise.json > <OUT>/Q2/bitwise.log 2>&1
```

Libraries that are not installed in `<E>` (TransformerEngine, Liger, SGLang) are reported as unavailable; that is fine. Install nothing.

**Expected:** `rl_kernel` and `diffusers` are batch-invariant; the `rl_kernel` forward equals the provider's bytes at every size. Report the timings as measured.

## What to send back

Push the branch `handoff/h100-h3-qk-rmsnorm-results-20261125` with `results/`: `Q1/tests.txt`, `Q2/`, `c-file.txt`, `nvidia-smi.txt` per task, `pip-list.txt` (torch, triton, diffusers, and any survey library), and `SUMMARY.md` with every deviation.
