# H100 handoff: GPU evidence for four H3 PRs that only ran on B200 (RFC #420), 2026-11-19

Four of the H3 PRs have GPU results from B200 only. This handoff runs the same commands on
H100 (sm90), so that every GPU claim in those PRs is measured on both architectures. Nothing
here changes code: each item runs one public fork branch at a fixed commit.

| Item | PR branch @ commit | What ran on B200 | GPUs |
|---|---|---|---|
| G1 | `feat/h3-fl2va-pipeline-contract` @ `df18d2d9`, `feat/h3-ref2va-pipeline-contract` @ `23dcb84b` | the rows' test files, including the CUDA transfer test | 1 |
| G2 | `fix/h3-ws2-test-doubles` @ `ba811b90`, against `test-h3` @ `10552d82` | `tests/distributed`; `tests/models/minimax_h3` with the strict build and pinned weights | 2 |
| G3 | `feat/h3-sglang-runtime-adapter` @ `8506f6a2` | the SGLang report at TP1/2/4/8 and the SGLang test files | as many as available, up to 8 |

The items are independent; run them in any order. Paths are placeholders: `<RK>` a clone of the
fork, `<OUT>` the results folder, `<PY>` the RL-Kernel Python (torch 2.13.0+cu130),
`<PROVIDER>` the Diffusers source at `f53d552036a0d1bd5570782a39cd40cfabf112bc`,
`<WEIGHTS>` the pinned H3 weight directory used for the earlier H3 H100 runs
(`RL_KERNEL_H3_WEIGHTS`), `<CKPT>` the released checkpoint at `42ed227e`.

```bash
git -C <RK> fetch https://github.com/fusheng-ji/RL-Kernel \
  fix/h3-ws2-test-doubles feat/h3-sglang-runtime-adapter \
  feat/h3-fl2va-pipeline-contract feat/h3-ref2va-pipeline-contract
for c in 10552d82 ba811b90 8506f6a2 df18d2d9 23dcb84b; do git -C <RK> worktree add --detach <WT>/$c $c; done
H=<this branch's checkout>/h100-handoff/gpu-gaps-20261119
```

Each worktree must stay clean: record `git -C <WT>/<c> status --porcelain --untracked-files=no`
next to each result. On shared storage that drops the executable bit, mode-only changes
(`100755 => 100644` in `git diff --summary`) are expected and harmless; note them.

For every item, also save `nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv`
taken just before it, as `<OUT>/<item>/gpu.txt`.

## G1. FL2VA and Ref2VA pipeline tests (1 H100, about 5 minutes)

On B200 these ran in two environments. Use the same two here.

```bash
for c in df18d2d9:fl2va 23dcb84b:ref2va; do
  sha=${c%%:*}; row=${c##*:}; cd <WT>/$sha
  files="tests/models/minimax_h3/test_h3_packed_layout.py tests/models/minimax_h3/test_h3_${row}_pipeline.py"
  # a) with Diffusers 0.43.0, SGLang 0.5.21, accelerate (and torchaudio for Ref2VA)
  PYTHONPATH=. <PY_PRIORART> -m pytest -q -p no:cacheprovider -rs $files > <OUT>/G1/$row-with-providers.txt 2>&1
  # b) the RL-Kernel environment, without Diffusers / SGLang
  PYTHONPATH=. <PY> -m pytest -q -p no:cacheprovider -rs $files > <OUT>/G1/$row-rlkernel.txt 2>&1
done
```

`<PY_PRIORART>` is the environment of G3 if it also has Diffusers 0.43.0 and accelerate;
otherwise say which packages were missing.

Expected (B200 numbers in brackets):

- FL2VA: a) all passed [43 passed]; b) only Diffusers / SGLang skips [36 passed, 7 skipped].
- Ref2VA: a) all passed [40 passed]; b) [36 passed, 4 skipped].
- `test_initial_rows_on_cuda_equal_cpu_bytes` passes in every run: on H100 the device copy of
  the layout and the initial rows is byte-equal to CPU, as on B200.

A skip for a missing package in a) is not a failure; list it.

## G2. Follow-up to #699 / #700 / #701 against `test-h3` (2 H100s)

The PR's claims are relative: against `test-h3`, `tests/distributed` loses its four
`_bootstrap` test-double failures, and `tests/models/minimax_h3` gains six passing tests and
no new failure. So both trees run the same suites in the same environment, and the result is
the difference. Absolute counts will differ from B200: on H100 the Hopper-gated rows run.

Strict build, in **each** of the two worktrees (`setup.py` builds for the visible GPU, so build
on the H100 node):

```bash
for c in 10552d82 ba811b90; do cd <WT>/$c
  RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 MAX_JOBS=8 <PY> setup.py build_ext --inplace > <OUT>/G2/build-$c.log 2>&1
  PYTHONPATH=. <PY> -c "import torch; from rl_engine.backends.extension import _C; \
print(torch.cuda.get_device_capability(0), _C.__file__)" > <OUT>/G2/env-$c.txt 2>&1
done
```

**`_C.__file__` must be inside that worktree** in both `env-*.txt` (an editable install of
another checkout can shadow it). If it is not, stop and send `G2/`.

Tests, with two GPUs visible (`CUDA_VISIBLE_DEVICES=<a>,<b>`):

```bash
for c in 10552d82 ba811b90; do cd <WT>/$c
  export PYTHONPATH=.:<PROVIDER>/src RL_KERNEL_H3_WEIGHTS=<WEIGHTS> HF_HUB_OFFLINE=1
  <PY> -m pytest -q -p no:cacheprovider --tb=no -rfs tests/distributed > <OUT>/G2/distributed-$c.txt 2>&1
  <PY> -m pytest -q -p no:cacheprovider --tb=short -rfs tests/models/minimax_h3 > <OUT>/G2/minimax_h3-$c.txt 2>&1
done
cd <OUT>/G2
for s in distributed minimax_h3; do for c in 10552d82 ba811b90; do
  grep -E '^(FAILED|ERROR)' $s-$c.txt | sed 's/ - .*//' | sort > $s-$c.failed; done
  { echo "== $s: failing only on ba811b90 (must be empty)"; comm -13 $s-10552d82.failed $s-ba811b90.failed
    echo "== $s: failing only on 10552d82"; comm -23 $s-10552d82.failed $s-ba811b90.failed; } >> diff.txt
done
```

The full `minimax_h3` suite with the pinned weights takes a while (on 2 x B200 it was the
longest run of the PR); run it in the background.

Expected:

- `distributed`: "failing only on 10552d82" lists exactly the four `_bootstrap` double tests:
  `test_run_world_reaps_workers[success]`, `[error]`, `[crash]` and
  `test_run_world_returns_rank_ordered_results`. "failing only on ba811b90" is empty.
  On B200, three tests failed on both trees (not H3: a ROCm `OpBackend` `KeyError`, the removed
  `KernelRegistry.get_loss_op`; and a CUDA IPC allocation of `test_h3_tp_adaln` on shared
  GPUs). Record which fail on H100.
- `minimax_h3`: "failing only on ba811b90" is empty, and `ba811b90` has 6 more passed tests
  than `10552d82` (the round-3 tests of `424f6279` and `390ac695`).
  On B200 one test failed on both trees,
  `test_h3_token_refiner.py::test_nodes_accuracy_repeat_batch_and_isolated[1-257]` (its
  `swiglu` bound); it is expected to **pass** on H100.

## G3. SGLang adapter (as many H100s as available, up to 8)

Environment, as on B200: torch 2.13.0+cu130 with SGLang **0.5.21** (`sglang-kernel` 0.4.7).
The adapter refuses any other SGLang version. The whole-model target JIT-builds SGLang kernels,
so `ninja` and `nvcc` (CUDA 13.0) must be on `PATH`. No RL-Kernel extension build is needed for
this item, but an existing `_C` does no harm.

```bash
cd <WT>/8506f6a2
export PYTHONPATH=. RL_KERNEL_H3_WEIGHTS=<WEIGHTS>
unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
<PY_SGLANG> tools/validation/models/h3_sglang_rollout_report.py --out <OUT>/G3/report.json \
  --tp <every size in 1 2 4 8 that fits the GPUs> > <OUT>/G3/report.stdout 2>&1; echo "report exit=$?"
<PY_SGLANG> -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3/test_h3_sglang_rollout.py \
  tests/models/minimax_h3/test_h3_checkpoint_mapper_vllm.py > <OUT>/G3/pytest.txt 2>&1
python $H/check_sglang_report.py <OUT>/G3/report.json $H/reference/sglang-report-b200-c9826852.json \
  > <OUT>/G3/check.txt; echo "check exit=$?"
<PY_SGLANG> tools/validation/models/plot_h3_sglang_rollout.py <OUT>/G3/report.json --out-dir <OUT>/G3
```

Expected:

- `check.txt` ends in `RESULT: PASS`: at every TP size, every bound output equals the
  training operator bitwise on every rank (and on a row prefix, where measured), equals TP1,
  and is restored on `remove()`; the whole-DiT readback is clean; the plugin hooks, installs,
  repeats, refuses Spectrum and catches the bypass.
- The native table in `check.txt` is informational. SGLang's own outputs differ from the
  training operator on B200 for `out_proj`, `fc2`, the AdaLN tables, the norms, the block
  output and the final layer. H100 values can differ from B200's; record them.
- `pytest.txt`: on 8 x B200, 55 passed and 2 skipped (vLLM-Omni not installed). With fewer
  GPUs, the TP sizes above the GPU count skip as well (`needs N GPUs`); nothing fails.

## Send back

`<OUT>/G1`, `<OUT>/G2` (logs, `*.failed`, `diff.txt`, `env-*.txt`, build logs) and `<OUT>/G3`
(report, stdout, pytest, check, charts and tables), each with its `gpu.txt` and the worktree
status. Strip host names, user names, job ids and paths before sending; keep hashes.

Push them as fork branch `handoff/h3-h100-gpu-gaps-results-20261119`, folder
`h100-results/gpu-gaps-20261119/`, as one signed-off commit on top of this branch. No PR and
no comment.

## Reference (`reference/`)

- `sglang-report-b200-c9826852.json`: the 8 x B200 report the SGLang PR quotes (identical to
  `pr_assets` `f7c40453`, `pending/h3-sglang-rollout/report.json`).
- `followup-test-counts-b200.json`: the B200 counts the follow-up PR quotes.
