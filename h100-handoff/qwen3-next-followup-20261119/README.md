# H100 handoff: Qwen3-Next follow-up branch (2026-11-19)

The whole Qwen3-Next stack (#503 … #551) is merged into `test-qwennext` (`bcc549bf`). This branch is a follow-up with five commits. It must run on B200 and on H100 before the PR is opened. B200 is being run separately.

Rules 1-7 of `h100-handoff/flink-round-20261116/README.md` still apply:
- clean detached tree;
- `rl_engine._C.__file__` checked before every GPU run;
- an **idle** node (the last round lost two attempts to another job's GPUs);
- everything redacted before it goes back.

## Code

The bundle in this directory, `qwen3-next-followup-20261119.bundle`, holds `feat/qwen3-next-followup-20261119` @ `8284ce2b`. Its prerequisite is `bcc549bf`, the head of upstream `RL-Align/RL-Kernel` `test-qwennext`.

```bash
cd <REPO>
git fetch https://github.com/RL-Align/RL-Kernel.git test-qwennext       # provides bcc549bf
git bundle verify <THIS_DIR>/qwen3-next-followup-20261119.bundle
git fetch <THIS_DIR>/qwen3-next-followup-20261119.bundle \
  'refs/heads/feat/qwen3-next-followup-20261119:refs/heads/feat/qwen3-next-followup-20261119'
git worktree add --detach <SRC> feat/qwen3-next-followup-20261119      # 8284ce2b
cd <SRC> && git config core.fileMode false && git status --porcelain --untracked-files=no   # must be empty
```

| Commit | Change | Covered by step |
|---|---|---|
| `ecd28a81` | CI: the four-GPU real-model step fails when a check is skipped | D |
| `add156f5` | the case recorder writes `xfail` / `xpass` instead of `skipped` | B (preflight operator cases), E |
| `e863c090` | `SharedQwenWorker` sets its trace attributes before `Worker.__init__` | B, C (every rollout), D |
| `55928823` | the environment identity refuses a user site that shadows an environment package | A, B, C |
| `8284ce2b` | every chain stage runs with `CUDA_DEVICE_MAX_CONNECTIONS=1` | B, C (`l2` recompute) |

## Environment

Use the environment of the last round (`handoff/h100-qwen3-next-results-20261118` @ `76079edc`):
- the base environment's apache-tvm-ffi 0.1.11;
- `CUDNN_HOME` and `LD_LIBRARY_PATH` pointed at the base environment's nvidia directories;
- `core.fileMode=false`.

There are two differences, and both are what this round tests:
- **Do not set `CUDA_DEVICE_MAX_CONNECTIONS`.** Run `unset CUDA_DEVICE_MAX_CONNECTIONS` in the launching shell. The launcher now sets it for every stage (commit `8284ce2b`). Last round it was exported by hand.
- **`PYTHONNOUSERSITE=1` for steps B-E.** Step A runs without it, on purpose.

## A. The user-site guard (no GPU, a few minutes)

```bash
cd <SRC>
unset PYTHONNOUSERSITE
python tools/validation/models/qwen3_next/qwen3_next_runtime.py --json > <OUT>/A/identity-usersite.json 2> <OUT>/A/identity-usersite.err; echo "rc=$?" > <OUT>/A/rc-usersite.txt
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage preflight --runs-root <OUT>/A/runs \
  --checkpoint <CKPT> --cuda-home <CUDA13> --python <PY> > <OUT>/A/preflight-usersite.log 2>&1; echo "rc=$?" >> <OUT>/A/rc-usersite.txt
PYTHONNOUSERSITE=1 python tools/validation/models/qwen3_next/qwen3_next_runtime.py --json > <OUT>/A/identity-nousersite.json; echo "rc=$?" > <OUT>/A/rc-nousersite.txt
```

**Expected.**
- Both commands without `PYTHONNOUSERSITE` exit non-zero.
- The error reads `environment_mismatch: the user site-packages shadows apache-tvm-ffi…` and may list more packages.
- The preflight attempt refuses before it uses any GPU.
- With `PYTHONNOUSERSITE=1` the identity prints and exits 0.
- If the user site holds no shadowing copy any more, write that down; then step A shows nothing.

## B. TP4 chain + full batch-invariance matrix (about 2.5 h)

```bash
export PYTHONNOUSERSITE=1; unset CUDA_DEVICE_MAX_CONNECTIONS
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --until l2 \
  --runs-root <OUT>/B/runs --checkpoint <CKPT> --cuda-home <CUDA13> --python <PY> > <OUT>/B/chain.log 2>&1
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage bi --bi-suite full --previous <OUT>/B/runs/<...-l2-...> \
  --runs-root <OUT>/B/runs --checkpoint <CKPT> --cuda-home <CUDA13> --python <PY> > <OUT>/B/bi.log 2>&1
python tools/validation/models/qwen3_next/qwen3_next_chain_report.py --last <OUT>/B/runs/<...-l2-...> --out <OUT>/B/chain-report.json > <OUT>/B/report.log 2>&1
bash ci/scripts/run_qwen3_next_chain_gate.sh <OUT>/B/chain-report.json > <OUT>/B/gate.log 2>&1; echo "gate rc=$?" >> <OUT>/B/gate.log
```

**Expected.**
- `preflight`, `layers`, `l2` and `bi` all pass, each run recording `source_commit` `8284ce2b…`.
- `l2` passes **without** the exported variable. Each stage's `attempts/0/` should show `CUDA_DEVICE_MAX_CONNECTIONS=1` if the stage logs its environment; say so if it does not.
- `bi-matrix.json` has 97 cells, all passed.
- The chain report says `passed=True` with 16 cells, and the gate exits 0.

## C. EP4 chain (about 1.2 h)

```bash
python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --until l2 --ep 4 \
  --runs-root <OUT>/C/runs --checkpoint <CKPT> --cuda-home <CUDA13> --python <PY> > <OUT>/C/chain.log 2>&1
```

**Expected.** `preflight`, `layers` and `l2` pass at `8284ce2b`.

## D. The two provider-GPU workflow steps, as CI runs them (about 15 min, four GPUs)

This runs the `run:` scripts of `.github/workflows/qwen3-next-provider-gpu.yml` itself, so the new skip guard is exercised.

```bash
cd <SRC>
python setup.py build_ext --inplace > <OUT>/D/build.log 2>&1
python -c "import rl_engine._C as c; print(c.__file__)" > <OUT>/D/c-file.txt     # must be under <SRC>
git status --porcelain --untracked-files=no                                    # still empty
python - <OUT>/D <<'PY'
import sys, yaml, pathlib
out = pathlib.Path(sys.argv[1])
steps = yaml.safe_load(open(".github/workflows/qwen3-next-provider-gpu.yml"))["jobs"]["provider"]["steps"]
for name, file in (("Check pinned providers in their own process", "provider.sh"),
                   ("Check the vLLM model contract and the EP4 real model (four GPUs)", "four-gpu.sh")):
    (out / file).write_text(next(s for s in steps if s.get("name") == name)["run"])
PY
for s in provider four-gpu; do
  RUNNER_TEMP=<OUT>/D TRITON_F32_DEFAULT=ieee PYTHONPATH=<SRC> bash --noprofile --norc -eo pipefail <OUT>/D/$s.sh > <OUT>/D/$s.log 2>&1
  echo "$s rc=$?" >> <OUT>/D/rc.txt
done
```

**Expected.**
- Both steps exit 0.
- The pytest summary in each log shows passes and **no** `skipped`.
- A non-zero exit with `::error::… was skipped` is a finding: send the log.

## E. CPU suite

```bash
PYTHONPATH=<SRC> python -m pytest -q -p no:cacheprovider tests/models/qwen3_next/ > <OUT>/E/pytest.log 2>&1; echo "rc=$?" >> <OUT>/E/pytest.log
```

**Expected.**
- No failures.
- B200 gave `264 passed, 108 skipped`. The count of skips may differ on H100.

## Send back

Push to a new fork branch `handoff/h100-qwen3-next-followup-results-20261119`, under `results/<A..E>/`. Use `git add -f` for `*.log`.

Include:
- the run directories, as in the last round;
- the chain report and gate log;
- `bi-matrix.json`;
- `nvidia-smi --query-gpu=name,driver_version --format=csv`;
- `pip list` filtered to torch / vllm / flashinfer / transformer_engine / megatron / apache-tvm-ffi;
- a `SUMMARY.md` with one row per step.

Keep failed attempts under `not-green-attempts/`, unedited. Redact host names, user names, paths, job ids and conda environment names before pushing.
