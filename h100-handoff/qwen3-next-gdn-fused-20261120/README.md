# H100 handoff: one-launch GDN recurrence for strict Qwen3-Next (2026-11-20)

## Why

The strict Qwen3-Next path ran each GDN layer's recurrence as a Python loop, launching vLLM's decode kernel once per token position. On a B200 that cost about 0.28 ms per token per layer. Across the 36 GDN layers that is about 10 of the ~11 ms per prompt token of strict prefill, and the same loop ran in VIME replay and in the training forward.

The fix, `63d46013` on branch `perf/qwen3-next-gdn-fused-recurrence`:
- **Forward:** one launch per layer. A small Triton kernel computes the gating values (`g`, `beta`) with the decode kernel's own expressions. vLLM 0.30.0's `fused_recurrent_gated_delta_rule` then runs the same per-token update with the same tiles, but with the time loop inside the kernel.
- **Backward:** it differentiates the old loop, now called `decode_loop_sequence`.
- **Provider name:** `GDNProviderConfig.recurrence` is renamed, so an older certificate does not cover the new code.

On B200 the new forward gives the same bits as the loop (output, final state, untouched cache slots) and the same gradients. This round checks the same on H100 (SM90), with real weights through the full chain, and measures the end-to-end speed.

Rules 1-7 of `h100-handoff/flink-round-20261116/README.md` still apply:
- clean detached tree;
- `rl_engine._C.__file__` checked before every GPU run;
- an **idle** node;
- everything redacted before it goes back.

**Idle node, again.** On B200, another job ending on the same node deleted vLLM's shared-memory buffers (`/dev/shm/psm_*`) and killed a run. Use a node, or an allocation, that no other job shares.

**Order relative to the follow-up handoff.** If `handoff/h100-qwen3-next-followup-20261119` (`8284ce2b`) has not been run yet, run it first: the follow-up PR cites that commit. This round's code contains those commits too, so the same environment works.

## Code

The bundle `qwen3-next-gdn-fused-20261120.bundle` in this directory holds two refs:

| Ref | Commit | Content |
|---|---|---|
| `perf/qwen3-next-gdn-fused-recurrence` | `63d46013` | follow-up `8284ce2b` + the fix |
| `local/e2e-bench-gdn-fused` | `213a9bcd` | the fix merged with the end-to-end speed test (`b6237c70`); **run everything from this one** |

Its prerequisite is `bcc549bf`, the head of upstream `test-qwennext`. Both refs are local only, not pushed.

```bash
cd <REPO>
git fetch https://github.com/RL-Align/RL-Kernel.git test-qwennext       # provides bcc549bf
git bundle verify <THIS_DIR>/qwen3-next-gdn-fused-20261120.bundle
git fetch <THIS_DIR>/qwen3-next-gdn-fused-20261120.bundle \
  'refs/heads/local/e2e-bench-gdn-fused:refs/heads/local/e2e-bench-gdn-fused'
git worktree add --detach <SRC> local/e2e-bench-gdn-fused             # 213a9bcd
cd <SRC> && git config core.fileMode false && git status --porcelain --untracked-files=no   # must be empty
```

## Environment

Use the follow-up round's environment:
- `PYTHONNOUSERSITE=1` for every step;
- `unset CUDA_DEVICE_MAX_CONNECTIONS` (the launcher sets it);
- `CUDNN_HOME` and `LD_LIBRARY_PATH` at the base environment's nvidia directories;
- `core.fileMode=false`.

**New:** before step B, install VIME's own pin of flash-linear-attention into the training environment. Only step D's stock-VIME arm needs it. VIME's Qwen3-Next model swallows the import error without it, then fails with `NameError: Qwen3NextAttention`.

```bash
<PY> -m pip install --no-deps flash-linear-attention==0.4.2 fla-core==0.4.2
```

Install it **before** step B, because the chain records the environment's fingerprint and the strict arm of step D checks it. If you prefer not to install it, step D's `replay native` rows fail; report that instead.

## A. One GPU: bits and per-layer time (about 20 min)

```bash
cd <SRC>
python setup.py build_ext --inplace > <OUT>/A/build.log 2>&1
python -c "import rl_engine._C as c; print(c.__file__)" > <OUT>/A/c-file.txt        # must be under <SRC>
git status --porcelain --untracked-files=no                                       # still empty
export PYTHONPATH=<SRC> TRITON_F32_DEFAULT=ieee
python <THIS_DIR>/gdn_one_launch_probe.py <SRC> > <OUT>/A/probe.jsonl 2> <OUT>/A/probe.log
python -m pytest -q -p no:cacheprovider \
  tests/models/qwen3_next/check_qwen3_next_gdn_sequence.py \
  tests/models/qwen3_next/check_qwen3_next_gdn_bridge.py \
  tests/models/qwen3_next/check_qwen3_next_shared_core.py \
  tests/models/qwen3_next/check_qwen3_next_core_matrix.py \
  tests/models/qwen3_next/check_qwen3_next_conv_bridge.py > <OUT>/A/checks.log 2>&1; echo "rc=$?" >> <OUT>/A/checks.log
```

**Expected.**
- Every line of `probe.jsonl` has `output_equal: true`, `state_equal: true` and `output_max_abs: 0.0`.
- `one_launch_ms` is far below `decode_loop_ms`. The B200 reference lines are below.
- `checks.log` shows 43 passed and rc=0, as on B200 (the 16 sequence checks are part of the 43).
- Any `false` in the probe, or any failing check, is the main finding of this round: send it as is.

B200 reference (`probe.jsonl`, same script):

| lengths | cache slots | equal | one launch | decode loop |
|---|---|---|---|---|
| 64 | 65 | yes / yes | 0.72 ms | 23.5 ms |
| 1024 | 65 | yes / yes | 2.68 ms | 380.6 ms |
| 1024, 777, 1500, 64 | 65 | yes / yes | 4.10 ms | 583.7 ms |
| 16 × 512 | 362 | yes / yes | 1.73 ms | 154.2 ms |

## B. TP4 chain + full batch-invariance matrix (about 2.5 h, four GPUs)

```bash
cd <SRC>; export PYTHONNOUSERSITE=1; unset CUDA_DEVICE_MAX_CONNECTIONS PYTHONPATH
R=tools/validation/models/qwen3_next/qwen3_next_run.py
python $R --stage all --until l2 --runs-root <OUT>/B/runs --checkpoint <CKPT> --cuda-home <CUDA13> --python <PY> > <OUT>/B/chain.log 2>&1
python $R --stage bi --bi-suite full --previous <OUT>/B/runs/<...-l2-...> --runs-root <OUT>/B/runs \
  --checkpoint <CKPT> --cuda-home <CUDA13> --python <PY> > <OUT>/B/bi.log 2>&1
python tools/validation/models/qwen3_next/qwen3_next_chain_report.py --last <OUT>/B/runs/<...-l2-...> --out <OUT>/B/chain-report.json > <OUT>/B/report.log 2>&1
PY=<PY> bash ci/scripts/run_qwen3_next_chain_gate.sh <OUT>/B/chain-report.json > <OUT>/B/gate.log 2>&1; echo "gate rc=$?" >> <OUT>/B/gate.log
```

Pass `PY=<PY>` to the gate. Without it the script uses whatever `python3` is on `PATH`, which on the B200 nodes had no pytest.

**Expected.**
- `preflight`, `layers`, `l2` and `bi` all pass, each run recording `source_commit` `213a9bcd…`.
- `bi-matrix.json` has 97 cells, all passed.
- The chain report says `passed=True` with 16 cells, and the gate exits 0.
- These stages are the real-weight evidence that the new forward did not change any bit the chain compares (rollout vs replay, batch invariance).
- The `l2` and `bi` stages should also be visibly shorter than in the follow-up round.

## C. EP4 chain (about 1.2 h)

```bash
python $R --stage all --until l2 --ep 4 --runs-root <OUT>/C/runs --checkpoint <CKPT> --cuda-home <CUDA13> --python <PY> > <OUT>/C/chain.log 2>&1
```

**Expected.** `preflight`, `layers` and `l2` pass at `213a9bcd`.

## D. End-to-end speed test, rollout and replay (about 3-4 h, four GPUs)

This runs on step B's `l2` run, in the same environment. Training is left out: at TP4 the actor needs about 120 GB per GPU.

```bash
cd <SRC>
python benchmarks/models/qwen3_next/qwen3_next_e2e_bench.py --run <OUT>/B/runs/<...-l2-...> \
  --out <OUT>/D/bench --phases rollout,replay > <OUT>/D/bench.log 2>&1; echo "rc=$?" >> <OUT>/D/bench.log
```

**What it measures.** Each arm runs in its own process on the same four GPUs, with the same synthetic inputs:

| Phase | Strict arm | Stock arms |
|---|---|---|
| rollout | the certified RL-Kernel vLLM engine | stock vLLM with CUDA graphs, eager, and with `VLLM_BATCH_INVARIANT=1` |
| replay | RL-Kernel model and logprob | VIME's own Qwen3-Next model |

The two workloads are the certified L2 shapes, plus 512-token prompts with 512 generated tokens.

**Expected.**
- `report.json` with `rl_kernel_commit` `213a9bcd…` and `tracked_tree_dirty: false`.
- Under `phases.rollout.unsupported`: the `native_bi` arm. vLLM 0.30.0 refuses it with "VLLM batch_invariant mode is not supported for GDN_ATTN". That is expected, not a failure.
- Every strict rollout shape has `same_tokens_across_repeats`, `same_tokens_across_rounds` and `row0_equals_batch1` all true, or null where they don't apply.
- Strict prefill well below the B200 numbers measured before the fix: 0.89 / 3.0 / 11.6 s for 64 / 256 / 1024-token prompts, and replay 1.0 / 2.8 / 9.6 s.
- Anything listed under `failures` is a finding: send its log from `bench/logs/`.

## E. CPU suite

```bash
PYTHONPATH=<SRC> python -m pytest -q -p no:cacheprovider tests/models/qwen3_next/ > <OUT>/E/pytest.log 2>&1; echo "rc=$?" >> <OUT>/E/pytest.log
```

**Expected.**
- No failures apart from one known group.
- In the training environment, 15 tests of `test_qwen3_next_training_checkpoint.py` may fail with `AssertionError: Invalid device id`. Their fixture patches `torch.cuda.device_count` to 0 before FlashInfer is first imported. This also happens on B200 at the follow-up commit, which did not touch the file.
- Report the count either way.

## Send back

Push to a new fork branch `handoff/h100-qwen3-next-gdn-fused-results-20261120`, under `results/<A..E>/`. Use `git add -f` for `*.log`.

Include:
- `probe.jsonl` and `checks.log`;
- the run directories, chain report and gate log, and `bi-matrix.json`;
- `bench/report.json` and `bench/logs/`;
- `nvidia-smi --query-gpu=name,driver_version --format=csv`;
- `pip list` filtered to torch / vllm / flashinfer / transformer_engine / megatron / flash-linear-attention / fla-core / apache-tvm-ffi;
- a `SUMMARY.md` with one row per step.

Keep failed attempts under `not-green-attempts/`, unedited. Redact host names, user names, paths, job ids and conda environment names before pushing. Do not open a PR or post comments.
