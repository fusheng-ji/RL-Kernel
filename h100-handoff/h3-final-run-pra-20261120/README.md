# H100 handoff: final_distributed_h3_run PR A on Hopper (2026-11-20)

Hardware: 2 nodes x 4 H100 (sm90); TP8 and TP4 x CP2 span both nodes. Paths are placeholders (`<REPO>`, `<SRC>`, `<BASE>`, `<OUT>`, `<THIS_DIR>`, `<E1>`, `<E2>`, `<VERL_OMNI>`, `<WEIGHTS>`, `<PROVIDER>`, `<OVERLAY>`).

## Why

PR A (RFC #420 row `final_distributed_h3_run`) adds the arithmetic the RL-Kernel arm of the final run needs outside the model:
- `exact_norm` / `clip_grad_norm_exact_`: an integer-binned gradient norm that is bit-identical across TP / CP layouts and ranks;
- `h3_sample_previous_step`: a strict drop-in for verl-omni's `sample_h3_transition` (CPS) whose log-prob does not depend on the batch;
- the NVRTC lookup for CUDA 13 wheels.

Everything below has run on one 8 x B200 node (reference numbers in each step). This round repeats it on H100, where the norm's binning kernel is compiled by NVRTC for sm90. The PR body will cite these results as its Hopper row.

## Rules (unchanged from the earlier H3 rounds)

- Clean detached tree; `git status --porcelain --untracked-files=no` empty before and after every step.
- `rl_engine._C.__file__` checked (inside `<SRC>`) before every GPU step that builds it.
- Idle nodes: no other job on the GPUs (the timing step especially). Record `nvidia-smi` at the start of each step.
- No push, no PR and no GitHub comment, except the one results branch at the end. A failure is reported as a failure, with its log; never change a script, a test or a bound to make a result pass.
- Anything sent back is redacted: no cluster, job, node, partition, user or personal path; hosts become `node-a` / `node-b`. Commits use `git commit -s` with no AI attribution lines.

## Code

`h3-final-run-pra-20261120.bundle` in this directory holds one ref, `feat/h3-final-run-transition-norm` at `70afabbd`. Its prerequisite is `12e81c36`, the head of upstream `test-h3`. The branch is local only, not pushed.

```bash
cd <REPO>
git fetch https://github.com/RL-Align/RL-Kernel.git test-h3              # provides 12e81c36
git bundle verify <THIS_DIR>/h3-final-run-pra-20261120.bundle
git fetch <THIS_DIR>/h3-final-run-pra-20261120.bundle \
  'refs/heads/feat/h3-final-run-transition-norm:refs/heads/feat/h3-final-run-transition-norm'
git worktree add --detach <SRC> feat/h3-final-run-transition-norm      # 70afabbd
git worktree add --detach <BASE> 12e81c36                              # test-h3, for the before / after rows
```

The probe scripts are in this directory: `compare_sample_h3_transition.py`, `logprob_batch_probe.py`, `norm_partition_probe.py`, `multirank_norm_probe.py`, `hook_before_after.py`, `timing_probe.py`. Run them from `<SRC>` with `PYTHONPATH=<SRC>` so that `rl_engine` is the checkout's.

## Environments

- **`<E1>`, the RL-Kernel H3 environment of the earlier rounds:** torch 2.13.0+cu130, the Diffusers provider pin `f53d552` (`<PROVIDER>/src`), the overlay with accelerate (`<OVERLAY>`), H3 weights (`RL_KERNEL_H3_WEIGHTS=<WEIGHTS>`).
- **`<E2>`, the final run's stack, only for the steps that import verl-omni:** torch 2.14.0+cu130, triton 3.8.0, diffusers 0.43.0, transformers 5.19.0, peft 0.22.0, accelerate 1.16.0, `verl @ git+https://github.com/verl-project/verl.git@2b4ed6826a39205838d0ca004bdd942334dbe4b6`, and verl-omni `a634f4d8` (`https://github.com/verl-project/verl-omni`) installed editable at `<VERL_OMNI>`. vLLM / vLLM-Omni are not needed for this round. If you build `<E2>` fresh, keep it apart from `<E1>`.

Record in each step's folder: `commit.txt`, `nvidia-smi.txt`, and `python -c "import torch, sys; print(torch.__version__, torch.version.cuda, sys.version)"`.

## A. One GPU: timing, norm partitions, Megatron hook (about 20 min)

Run the timing first, on an idle GPU.

```bash
cd <SRC>; export PYTHONPATH=<SRC>
CUDA_VISIBLE_DEVICES=0 <E2>/bin/python <THIS_DIR>/timing_probe.py --out <OUT>/A/timing.json --repeats 30 > <OUT>/A/timing.log 2>&1
CUDA_VISIBLE_DEVICES=0 <E1>/bin/python <THIS_DIR>/norm_partition_probe.py --out <OUT>/A/norm-partition.json > <OUT>/A/norm-partition.log 2>&1
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29811 <E1>/bin/python <THIS_DIR>/hook_before_after.py \
  <BASE>/rl_engine/ops/norm/reproducible.py <SRC>/rl_engine/ops/norm/reproducible.py <OUT>/A/megatron-hook.json > <OUT>/A/megatron-hook.log 2>&1
```

**Expected (B200 reference, same scripts):**
- `megatron-hook.json`: `all_equal: true` on the 3 cases. The `test-h3` side needs `libnvrtc.so.12` on the loader path; if it fails with `libnvrtc.so.12: cannot open shared object file`, that is the bug PR A fixes: report it, and say which NVRTC files the environment has.
- `norm-partition.json`: `exact_bitwise_equal` 16 / 16. B200 gave `torch_bitwise_equal` 15 / 16 (the 4-shard original order differs by one FP32 ulp); on H100 report whatever torch gives.
- `timing.json`, B200 medians:

  | Call | torch / verl-omni (ms) | RL-Kernel (ms) |
  |---|---|---|
  | norm | 1.45 | 8.99 |
  | norm + clip | 1.91 | 9.54 |
  | transition, video, sampling | 0.38 | 97.50 |
  | transition, video, scoring | 0.34 | 97.43 |
  | transition, audio, sampling | 0.31 | 0.41 |
  | transition, audio, scoring | 0.28 | 0.39 |

## B. verl-omni comparisons (`<E2>`, one GPU, about 15 min)

```bash
cd <SRC>; export PYTHONPATH=<SRC>
CUDA_VISIBLE_DEVICES=0 <E2>/bin/python <THIS_DIR>/compare_sample_h3_transition.py --out <OUT>/B/verl-omni-compare.json > <OUT>/B/verl-omni-compare.log 2>&1
CUDA_VISIBLE_DEVICES=0 <E2>/bin/python <THIS_DIR>/logprob_batch_probe.py --out <OUT>/B/logprob-batch.json > <OUT>/B/logprob-batch.log 2>&1
CUDA_VISIBLE_DEVICES=0 VERL_OMNI_PATH=<VERL_OMNI> <E2>/bin/python -m pytest \
  "tests/models/minimax_h3/test_h3_runtime_integration.py::TestTransition" -p no:cacheprovider -q -rfs > <OUT>/B/transition-tests.txt 2>&1
```

**Expected:** `verl-omni-compare.json` has 560 cases with `prev_sample`, `prev_sample_mean`, `std_dev_t` and `sqrt_dt` all 560 / 560; `log_prob` is not expected to be equal (B200: 8 / 560, max abs difference 6.3e-06). `logprob-batch.json` has `rl_kernel_differ: 0` (B200: `verl_omni_differ: 27` of 144). `transition-tests.txt`: 9 passed. Any RL-Kernel `log_prob` that differs from its batch-of-1 value, or any non-log-prob output that differs from verl-omni, is the main finding: send it as is.

## C. Multi-GPU norm (`<E1>`, about 15 min)

One node, then both nodes (TP8 and TP4 x CP2):

```bash
cd <SRC>; export PYTHONPATH=<SRC>
for n in 2 4; do
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n-1))) <E1>/bin/torchrun --nproc-per-node $n --master-port $((29800+n)) \
    <THIS_DIR>/multirank_norm_probe.py --out <OUT>/C/multirank-$n.json > <OUT>/C/multirank-$n.log 2>&1
done
# on node-a and node-b, the same command with --node-rank 0 / 1:
<E1>/bin/torchrun --nnodes 2 --node-rank <0|1> --nproc-per-node 4 --master-addr <node-a address> --master-port 29808 \
  <THIS_DIR>/multirank_norm_probe.py --out <OUT>/C/multirank-8.json > <OUT>/C/multirank-8-<node>.log 2>&1
```

The 8-rank run writes the report on global rank 0 (node-a).

**Expected (B200: all true):** for TP2, TP4, TP8 and TP4 x CP2, `exact_norm_equal_all_ranks` and `exact_clipped_equal` true and `exact_norm_values` one value, equal to `one_gpu.exact_norm`. Report the torch columns as they come: B200 gave the one-GPU norm and clipped bytes on all four layouts.

## D. Test suite, head vs `test-h3` (`<E1>`, one node, 2 + 2 GPUs, about 25 min)

```bash
for side in head base; do W=<SRC>; [ $side = base ] && W=<BASE>
  (cd $W && RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E1>/bin/python setup.py build_ext --inplace > <OUT>/D/build-$side.log 2>&1)
  (cd $W && PYTHONPATH=$W <E1>/bin/python -c "import rl_engine._C as c; print(c.__file__)") > <OUT>/D/c-file-$side.txt   # inside $W
done
(cd <SRC> && CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=<SRC>:<PROVIDER>/src:<OVERLAY> <E1>/bin/python -m pytest -q -p no:cacheprovider --tb=line -rfs tests/models/minimax_h3 tests/ops/norm > <OUT>/D/suite-head.txt 2>&1) &
(cd <BASE> && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=<BASE>:<PROVIDER>/src:<OVERLAY> <E1>/bin/python -m pytest -q -p no:cacheprovider --tb=line -rfs tests/models/minimax_h3 tests/ops/norm > <OUT>/D/suite-base.txt 2>&1) &
wait
```

**Expected:** nothing fails only at head. B200: head 1562 passed / 2 failed / 148 skipped, `test-h3` 1539 / 2 / 148; the 2 failures on both are `test_h3_token_refiner.py::test_nodes_accuracy_repeat_batch_and_isolated[1-257]` (a B200 bound; it passed on H100 in earlier rounds) and `test_rms_norm.py::test_registry_dispatches_rms_norm`. List every failure on each side.

## What to send back

One fork branch, `handoff/h100-h3-final-run-pra-results-20261120`, cut from upstream `test-h3` (`12e81c36`), holding `results/` with:
- `A/`, `B/`, `C/`, `D/`: the outputs above, each with `commit.txt`, `nvidia-smi.txt` and the environment line;
- `SUMMARY.md`: per step, the numbers asked for and every deviation from the expectation, and the H100 timing table in the same form as the B200 one;
- `pip-list-filtered.txt` for `<E1>` and `<E2>` (torch, triton, diffusers, transformers, verl, verl-omni, peft, accelerate, nvidia-*).

Push only that branch. Run the redaction scan (hostnames other than `node-a` / `node-b`, user, cluster, partition, job id, `/home/…`, `/data/…`, email) over everything in it first. The commit is `git commit -s` with no AI attribution.
