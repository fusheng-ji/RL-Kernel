# H100 handoff: stock VIME replay at its recipe layout (2026-11-21)

## Why

In the last round (`handoff/h100-qwen3-next-gdn-fused-20261120`, results `4150731a`), step D's stock VIME replay arm failed: Megatron rejected the expert weights' shape. Stock VIME cannot run Qwen3-Next at TP4 at all:
- **Expert weights.** VIME's loader only shards expert weights for expert tensor parallel 1. Megatron's grouped expert layers carry no tensor-parallel flag, so at ETP4 the loader copies the whole expert tensor.
- **Attention output gate.** Qwen3-Next has 2 KV heads. When TP exceeds that, Megatron gives each rank the query heads of a whole KV group and keeps only its share for attention, but it does not narrow the attention output gate. The forward then fails on the gate's shape, at any EP.

So the speed test now runs stock VIME at its own recipe's layout (`scripts/run-qwen3-next-80B-A3B.sh`), on the same four GPUs:
- TP2 with sequence parallel. Without sequence parallel, Megatron's MoE layer refuses to train at TP > 1.
- Expert tensor parallel 1, with the experts spread over all four ranks (EP4).
- Data parallel 2. The two replicas split each step's samples. A single sample leaves one replica a filler copy, because every rank must take part in the MoE all-to-all. The step time is the slowest rank's.

The strict arm keeps its certified TP4 profile without sequence parallel. Each arm's layout is recorded in `report.json` under `phases.replay.layouts`.

| Commit | Content |
|---|---|
| `64f18040` | stock VIME at TP2 / EP4 / ETP1 / DP2 |
| `e0f1959c` | stock VIME with sequence parallel; replay pads packed tokens to a multiple of TP the way VIME's `get_batch` does (no current shape needs padding) |
| `37e04258` | both merged into the GDN fix + speed test of last round (`213a9bcd`); **run everything from this one** |

On B200 the same steps passed at this commit (2026-11-21):
- stock VIME replay and training at this layout, run first as a smoke test;
- `preflight`, `layers`, `l2` and `bi` at `37e04258`, with bi 97/97 and the gate exiting 0;
- the whole speed test (rollout, replay, train) with no failures.

The B200 replay numbers are in step C.

Rules 1-7 of `h100-handoff/flink-round-20261116/README.md` still apply:
- clean detached tree;
- `rl_engine._C.__file__` checked before every GPU run;
- an **idle** node (vLLM's `/dev/shm` buffers are deleted when another job on the node ends);
- everything redacted before it goes back.

## Code

The bundle `qwen3-next-stock-tp2-20261121.bundle` in this directory holds two refs:

| Ref | Commit |
|---|---|
| `local/e2e-bench-gdn-fused-tp2` | `37e04258`, the one to run |
| `feat/qwen3-next-e2e-bench` | `e0f1959c`, the speed-test branch alone (for reference) |

Its prerequisite is `bcc549bf`, the head of upstream `test-qwennext`. Both refs are local only, not pushed.

```bash
cd <REPO>
git fetch https://github.com/RL-Align/RL-Kernel.git test-qwennext       # provides bcc549bf
git bundle verify <THIS_DIR>/qwen3-next-stock-tp2-20261121.bundle
git fetch <THIS_DIR>/qwen3-next-stock-tp2-20261121.bundle \
  'refs/heads/local/e2e-bench-gdn-fused-tp2:refs/heads/local/e2e-bench-gdn-fused-tp2'
git worktree add --detach <SRC> local/e2e-bench-gdn-fused-tp2         # 37e04258
cd <SRC> && git config core.fileMode false && git status --porcelain --untracked-files=no   # must be empty
```

## Environment

Use the last round's environment unchanged:
- the training environment, with `flash-linear-attention==0.4.2` and `fla-core==0.4.2` already installed;
- `PYTHONNOUSERSITE=1` for every step;
- `unset CUDA_DEVICE_MAX_CONNECTIONS` (the launcher sets it);
- `CUDNN_HOME` and `LD_LIBRARY_PATH` at the base environment's nvidia directories.

Install nothing new. The chain records the environment's fingerprint, and the strict arm of step C checks it.

## A. Build (about 5 min)

```bash
cd <SRC>
python setup.py build_ext --inplace > <OUT>/A/build.log 2>&1
python -c "import rl_engine._C as c; print(c.__file__)" > <OUT>/A/c-file.txt        # must be under <SRC>
git status --porcelain --untracked-files=no                                       # still empty
```

## B. TP4 chain + full batch-invariance matrix (about 2.5 h, four GPUs)

`37e04258` changes the VIME actor runtime file, so the strict arm needs a fresh `l2` run at this commit.

```bash
cd <SRC>; export PYTHONNOUSERSITE=1; unset CUDA_DEVICE_MAX_CONNECTIONS PYTHONPATH
R=tools/validation/models/qwen3_next/qwen3_next_run.py
python $R --stage all --until l2 --runs-root <OUT>/B/runs --checkpoint <CKPT> --cuda-home <CUDA13> --python <PY> > <OUT>/B/chain.log 2>&1
python $R --stage bi --bi-suite full --previous <OUT>/B/runs/<...-l2-...> --runs-root <OUT>/B/runs \
  --checkpoint <CKPT> --cuda-home <CUDA13> --python <PY> > <OUT>/B/bi.log 2>&1
python tools/validation/models/qwen3_next/qwen3_next_chain_report.py --last <OUT>/B/runs/<...-l2-...> --out <OUT>/B/chain-report.json > <OUT>/B/report.log 2>&1
PY=<PY> bash ci/scripts/run_qwen3_next_chain_gate.sh <OUT>/B/chain-report.json > <OUT>/B/gate.log 2>&1; echo "gate rc=$?" >> <OUT>/B/gate.log
```

**Expected.**
- `preflight`, `layers`, `l2` and `bi` all pass, each run recording `source_commit` `37e04258…`.
- `bi-matrix.json` has 97 cells, all passed.
- The chain report says `passed=True` with 16 cells, and the gate exits 0.
- For the strict path this commit only changes how the stock arm is built, so these are expected to match last round's B exactly.

The EP4 chain is not rerun. The bench runs on the TP4 `l2` run, and the strict EP4 path is unchanged since `213a9bcd`, which passed last round's C.

## C. End-to-end speed test, rollout and replay (about 3-4 h, four GPUs)

This runs on step B's `l2` run, in the same environment. Training is left out, as last round: the strict TP4 actor needs about 120 GB per GPU.

```bash
cd <SRC>
python benchmarks/models/qwen3_next/qwen3_next_e2e_bench.py --run <OUT>/B/runs/<...-l2-...> \
  --out <OUT>/C/bench --phases rollout,replay > <OUT>/C/bench.log 2>&1; echo "rc=$?" >> <OUT>/C/bench.log
```

**Expected.**
- `rc=0`, and `report.json` with `rl_kernel_commit` `37e04258…` and `tracked_tree_dirty: false`.
- `phases.replay.failures` absent or empty. Both `strict` and `native` replay results exist for both workloads, together with `strict_over_native`.
- `phases.replay.layouts`:
  - `strict`: `{"tp": 4, "sp": false, "ep": 1, "etp": 4, "dp": 1}`;
  - `native`: `{"tp": 2, "sp": true, "ep": 4, "etp": 1, "dp": 2}`.
- `phases.rollout.unsupported` lists only the `native_bi` arm ("VLLM batch_invariant mode is not supported for GDN_ATTN"), as last round.
- Every strict rollout shape has `same_tokens_across_repeats`, `same_tokens_across_rounds` and `row0_equals_batch1` all true, or null where they don't apply.
- Strict replay and rollout times are close to last round's. Strict replay medians were, for p64 / p256 / p1024: 0.29 / 0.29 / 0.42 s at b1, and 0.31 / 0.32 / 0.57 s at b4.
- Anything listed under `failures` is a finding: send its log from `C/bench/logs/`.

B200 reference (same commit, job of 2026-11-21), replay medians in seconds:

| workload | shape | strict (TP4) | stock VIME (TP2 + SP, DP2) | strict / stock |
|---|---|---|---|---|
| certified | p64-b1 | 0.200 | 0.265 | 0.75 |
| certified | p64-b4 | 0.201 | 0.292 | 0.69 |
| certified | p256-b1 | 0.202 | 0.286 | 0.71 |
| certified | p256-b4 | 0.211 | 0.302 | 0.70 |
| certified | p1024-b1 | 0.293 | 0.296 | 0.99 |
| certified | p1024-b4 | 0.432 | 0.308 | 1.40 |
| long | p512-b1-r512 | 0.297 | 0.287 | 1.03 |
| long | p512-b4-r512 | 0.437 | 0.312 | 1.40 |
| long | p512-b16-r512 | 1.146 | 0.338 | 3.39 |

H100 numbers will differ; what matters is that both arms run and that the strict arm stays close to last round's H100 values.

## D. CPU suite

```bash
PYTHONPATH=<SRC> python -m pytest -q -p no:cacheprovider tests/models/qwen3_next/ > <OUT>/D/pytest.log 2>&1; echo "rc=$?" >> <OUT>/D/pytest.log
```

**Expected.**
- No failures apart from the 15 known `AssertionError: Invalid device id` failures of `test_qwen3_next_training_checkpoint.py`. Its fixture patches `torch.cuda.device_count` to 0 before FlashInfer is first imported; the file is untouched.
- Report the count either way. Last round: 15 failed, 658 passed. This round adds 5 tests.

## Send back

Push to a new fork branch `handoff/h100-qwen3-next-stock-tp2-results-20261121`, under `results/<A..D>/`. Use `git add -f` for `*.log`.

Include:
- `c-file.txt`;
- the run directories' `run.json` and `result.json`, the chain report, the gate log, and `bi-matrix.json`;
- `C/bench/report.json` and `C/bench/logs/`;
- `nvidia-smi --query-gpu=name,driver_version --format=csv`;
- `pip list` filtered to torch / vllm / flashinfer / transformer_engine / megatron / flash-linear-attention / fla-core / apache-tvm-ffi;
- a `SUMMARY.md` with one row per step.

Keep failed attempts under `not-green-attempts/`, unedited. Redact host names, user names, paths, job ids and conda environment names before pushing. Do not open a PR or post comments.
