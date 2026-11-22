# H100 handoff: stock vLLM rollout at the CUDA default connection limit (2026-11-23)

## Why

In the speed test of #745 (last round: `handoff/h100-qwen3-next-stock-tp2-20261121`, results `e150b4b4`), every arm ran with the stage environment's `CUDA_DEVICE_MAX_CONNECTIONS=1`, the stock vLLM arms included. The chain sets it because stock Megatron refuses TP > 1 without it, but vLLM does not need it. So the stock vLLM baseline was not quite vLLM as it ships.

`3e06906f` drops it for the stock vLLM rollout arms only:
- the stock VIME arm keeps it (Megatron needs it);
- the strict arm keeps its certified stage environment.

This round re-measures the rollout phase at that commit. Replay and training are unchanged and are not rerun.

The same run is going on B200 at the same commit.

Rules 1-7 of `h100-handoff/flink-round-20261116/README.md` still apply:
- clean detached tree;
- `rl_engine._C.__file__` checked before every GPU run;
- an **exclusive** node (vLLM's `/dev/shm` buffers are deleted when another job on the node ends);
- everything redacted before it goes back.

## Code

The bundle `qwen3-next-rollout-default-20261123.bundle` in this directory holds one ref, `local/rollout-default` at `3e06906f`. Its prerequisite is `8fbacb87`, the head of #745 on the fork.

```bash
cd <REPO>
git fetch https://github.com/fusheng-ji/RL-Kernel.git feat/qwen3-next-e2e-bench     # provides 8fbacb87
git bundle verify <THIS_DIR>/qwen3-next-rollout-default-20261123.bundle
git fetch <THIS_DIR>/qwen3-next-rollout-default-20261123.bundle \
  'refs/heads/local/rollout-default:refs/heads/local/rollout-default'
git worktree add --detach <SRC> local/rollout-default                                # 3e06906f
cd <SRC> && git config core.fileMode false && git status --porcelain --untracked-files=no   # must be empty
```

## Environment

Use last round's environment unchanged:
- the training environment, with `flash-linear-attention==0.4.2` and `fla-core==0.4.2`;
- `PYTHONNOUSERSITE=1` for every step;
- `unset CUDA_DEVICE_MAX_CONNECTIONS` (the launcher sets it for the stages);
- `CUDNN_HOME` and `LD_LIBRARY_PATH` at the base environment's nvidia directories.

Install nothing new.

## A. Build (about 5 min)

```bash
cd <SRC>
python setup.py build_ext --inplace > <OUT>/A/build.log 2>&1
python -c "import rl_engine._C as c; print(c.__file__)" > <OUT>/A/c-file.txt        # must be under <SRC>
git status --porcelain --untracked-files=no                                       # still empty
```

## B. TP4 chain to `l2` (about 1 h, four GPUs)

The speed test runs from the `l2` run's own source clone, so it needs an `l2` run at this commit. `bi` is not needed this round.

```bash
cd <SRC>; export PYTHONNOUSERSITE=1; unset CUDA_DEVICE_MAX_CONNECTIONS PYTHONPATH
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --until l2 --runs-root <OUT>/B/runs \
  --checkpoint <CKPT> --cuda-home <CUDA13> --python <PY> > <OUT>/B/chain.log 2>&1
```

**Expected:** `preflight`, `layers` and `l2` pass, each run recording `source_commit` `3e06906f…`.

## C. Speed test, rollout only (about 2 h, four GPUs)

```bash
cd <SRC>
python benchmarks/models/qwen3_next/qwen3_next_e2e_bench.py --run <OUT>/B/runs/<...-l2-...> \
  --out <OUT>/C/bench --phases rollout > <OUT>/C/bench.log 2>&1; echo "rc=$?" >> <OUT>/C/bench.log
```

**Expected:**
- `rc=0`; `report.json` with `rl_kernel_commit` `3e06906f…` and `tracked_tree_dirty: false`.
- `phases.rollout.unsupported` lists only `native_bi` for both workloads, as before.
- Every strict rollout shape has `same_tokens_across_repeats`, `same_tokens_across_rounds` and `row0_equals_batch1` true, or null where they do not apply.
- Strict rollout times close to last round's at `37e04258`, since the strict arm's environment is unchanged.
- **What this round measures:** the stock vLLM arms (`native_graph`, `native_eager`) without the limit. Last round's H100 totals, with `CUDA_DEVICE_MAX_CONNECTIONS=1`, in seconds:

| Workload | Shape | strict | stock, CUDA graphs | stock, eager |
|---|---|---:|---:|---:|
| certified | p64 b1 | 4.52 | 0.10 | 1.75 |
| certified | p1024 b4 | 5.57 | 0.31 | 1.97 |
| long | p512 b1 | 163.2 | 2.26 | 54.85 |
| long | p512 b16 | 217.2 | 3.50 | 56.25 |
| long | p512 b64 | 370.5 | 5.93 | 57.06 |

Any change in the stock columns is the finding; report it either way.

## Send back

Push to a new fork branch `handoff/h100-qwen3-next-rollout-default-results-20261123`, under `results/<A..C>/`. Use `git add -f` for `*.log`.

Include:
- `c-file.txt`;
- the run directories' `run.json` and `result.json`;
- `C/bench/report.json` and `C/bench/logs/`;
- `nvidia-smi --query-gpu=name,driver_version --format=csv`;
- a `SUMMARY.md` with one row per step.

Keep failed attempts under `not-green-attempts/`, unedited. Redact host names, user names, paths, job ids and environment names before pushing. Do not open a PR or post comments.
