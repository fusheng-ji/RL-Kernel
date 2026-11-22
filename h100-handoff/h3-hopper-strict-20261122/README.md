# H100 handoff: H3 strict-boundary runs of #732, #733 and #735 (2026-11-22)

Two independent tasks. On B200 these PRs ran the 8 boundary modules (patch and audio projections, text projection, token refiner, time embedder, final norm, heads) on the provider (`--provider-boundaries` / `strict_boundaries=False`), because the strict video projection rows need SM90 (#646). The 50 blocks are strict in every run. These runs repeat the B200 measurements with **every boundary module strict**, which only Hopper can do; the PRs list them as their open Hopper item.

| Task | PRs | Hardware | Environment | Time (est.) |
|---|---|---|---|---|
| **S1** | #732 weight-sharded model, #733 LoRA on the grid | 2 nodes x 4 H100; TP8 and TP4 x CP2 span both | `<E1>` | about 2.5 h |
| **S2** | #735 vLLM-Omni TP4 rollout binding | 1 node x 4 H100, exclusive | `<E2>` + vLLM + vLLM-Omni | about 1 h |

Paths are placeholders: `<REPO>` (a clone of the fork), `<THIS_DIR>` (this folder), `<OUT>`, `<CKPT>` (the H3 checkpoint dir with `transformer/` and `FL2VA/`, revision `42ed227ee7df40d41602854ae760620d6eb651fe`), `<E1>`, `<E2>`, `<PROVIDER>`, `<OVERLAY>`, `<VERL_OMNI>`.

## Rules (unchanged from the earlier H3 rounds)

- Detached worktrees; `git status --porcelain --untracked-files=no` empty before and after every step. Change no file.
- `rl_engine._C.__file__` checked (inside the checkout) before every GPU step.
- Idle nodes, nothing else on the GPUs; record `nvidia-smi` at the start of each step.
- A failure is reported as a failure, with its log. Never change a script, a test or a bound to make a result pass; failed attempts go, as they are, into `not-green-attempts/`.
- Push nothing but the results branch at the end. No PR, no comment.
- Redact everything sent back: no cluster, job, node, partition, user, conda environment name, `/home/…` or `/data/…` path, email; hosts become `node-a` / `node-b`. `git add -f` the `*.log` files. Commit with `git commit -s`, no AI attribution lines.

## Launching multi-rank tools

`h3_parallel_full_model.py` reads the SLURM rank variables. Under `srun` (one task per GPU) run it directly. Under `torchrun`, go through `<THIS_DIR>/slurm_env.py`, which maps `RANK` / `WORLD_SIZE` / `LOCAL_RANK` onto them and moves the tool's own rendezvous one port above torchrun's store:

```bash
# one node, n GPUs
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n-1))) <E1>/bin/torchrun --nproc-per-node $n --master-port 29700 \
  <THIS_DIR>/slurm_env.py tools/validation/models/h3_parallel_full_model.py <flags>
# two nodes (8 ranks), on node-a and node-b with --node-rank 0 / 1
<E1>/bin/torchrun --nnodes 2 --node-rank <0|1> --nproc-per-node 4 --master-addr <node-a address> --master-port 29710 \
  <THIS_DIR>/slurm_env.py tools/validation/models/h3_parallel_full_model.py <flags> --local-size 4
```

`--local-size 4` (GPUs per node) is needed whenever a TP group spans the nodes (TP8): it selects the hierarchical collective. Global rank 0 (node-a) writes `report.json`. Use a new port for every run.

---

## S1. #732 and #733, strict boundaries (2 x 4 H100, `<E1>`)

`<E1>` is the RL-Kernel H3 environment of the earlier rounds: torch 2.13.0+cu130, the Diffusers provider pin `f53d552` (`<PROVIDER>/src`), the overlay with accelerate (`<OVERLAY>`).

### Code

Both PRs are public; take their heads:

```bash
cd <REPO>
git fetch https://github.com/RL-Align/RL-Kernel.git pull/732/head pull/733/head
git worktree add --detach <S732> 6c7e424c   # #732 head (feat/h3-sharded-model)
git worktree add --detach <S733> 4e67d5d7   # #733 head (feat/h3-parallel-lora; contains #732)
```

Build `_C` in each (`RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E1>/bin/python setup.py build_ext --inplace`) and check `_C.__file__`. Every run below uses, from its checkout:

```bash
export PYTHONPATH=<S73x>:<PROVIDER>/src:<OVERLAY>
COMMON="--checkpoint <CKPT> --attention h3_full --out <OUT>/S1/<run>"
```

and **no** `--provider-boundaries` (that is the point of this task).

### S1.a #732 at `6c7e424c`

| Run | Ranks | Flags (after `$COMMON`) |
|---|---|---|
| `harness_tp4_sp` (control: full weights) | 4 | `--tp 4 --sp --size 928` |
| `sharded_tp2_40` | 2 | `--tp 2 --sharded --size 40` |
| `sharded_tp4_sp_928` | 4 | `--tp 4 --sp --sharded --size 928` |
| `sharded_tp4_cp2_ring_925` | 8 (2 nodes) | `--tp 4 --cp 2 --cp-mode ring --sharded --size 925` |
| `sharded_tp4_cp2_sp_ring_928` | 8 | `--tp 4 --cp 2 --sp --cp-mode ring --sharded --size 928` |
| `sharded_tp4_cp2_sp_ulysses_928` | 8 | `--tp 4 --cp 2 --sp --cp-mode ulysses --sharded --size 928` |
| `sharded_tp8_sp_928` | 8 | `--tp 8 --sp --sharded --size 928 --local-size 4` |
| `inject_sharded_tp4_sp_928` (negative control) | 4 | `--tp 4 --sp --sharded --size 928 --inject 17:700:2` |
| `inject_sharded_tp4_cp2_ring_925` (negative control) | 8 | `--tp 4 --cp 2 --cp-mode ring --sharded --size 925 --inject 17:5:3` |

### S1.b #733 at `4e67d5d7`, with LoRA

Add `--sharded --lora-rank 64 --lora-alpha 128 --lora-seed 7` to every run; each grid runs twice, with B random (no extra flag) and with `--lora-b-zero`.

| Run | Ranks | Flags |
|---|---|---|
| `lora_{random,zero}_tp2_40` | 2 | `--tp 2 --size 40` |
| `lora_{random,zero}_tp4_sp_928` | 4 | `--tp 4 --sp --size 928` |
| `lora_{random,zero}_tp4_cp2_ring_925` | 8 | `--tp 4 --cp 2 --cp-mode ring --size 925` |
| `lora_{random,zero}_tp4_cp2_sp_ring_928` | 8 | `--tp 4 --cp 2 --sp --cp-mode ring --size 928` |
| `lora_{random,zero}_tp4_cp2_sp_ulysses_928` | 8 | `--tp 4 --cp 2 --sp --cp-mode ulysses --size 928` |
| `lora_{random,zero}_tp8_sp_928` | 8 | `--tp 8 --sp --size 928 --local-size 4` |
| `inject_lora_tp4_sp_928` | 4 | `--tp 4 --sp --size 928 --inject 17:700:2` (B random) |
| `inject_lora_tp4_cp2_ring_925` | 8 | `--tp 4 --cp 2 --cp-mode ring --size 925 --inject 17:5:3` (B random) |

### S1.c Test suites (one node, 2 + 2 GPUs, about 30 min)

```bash
(cd <S733> && CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=<S733>:<PROVIDER>/src:<OVERLAY> <E1>/bin/python -m pytest -q -p no:cacheprovider --tb=line -rfs tests/models/minimax_h3 > <OUT>/S1/suite-733.txt 2>&1) &
(cd <S732> && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=<S732>:<PROVIDER>/src:<OVERLAY> <E1>/bin/python -m pytest -q -p no:cacheprovider --tb=line -rfs tests/models/minimax_h3 > <OUT>/S1/suite-732.txt 2>&1) &
wait
```

### S1 expected

- **Every non-inject run:** `all_bitwise: true` in `report.json`, every rank's `checks` true (video and audio velocity, input gradients, every block output and gradient, every weight-shard gradient, and for #733 every LoRA gradient), `shards_missing` empty (#733's tool). `strict_boundaries: true` in each report. B200 (provider boundaries) was bitwise on every one of these grids. A strict-boundary run that is not bitwise is the main finding: send it as it is, with `first_drift_forward` / `first_drift_backward`.
- **Inject runs:** not bitwise, and `first_drift_forward` on every rank names block 17 and the row: `17:700:2` → row 700 (video), 1 row, owner rank [2]; `17:5:3` → row 5 (text), 1 row, owners [0, 1, 2, 3]. For #733, `lora_gradients_bitwise: false` on every rank.
- **Memory (record `resident_param_bytes` and the peak per rank; B200 values with provider boundaries):** weights per rank 43.78 GiB (TP2), 34.81 GiB (TP4), 30.33 GiB (TP8); grid peak 45.35 / 37.78 / 33.03 GiB (#732), LoRA 415.6 / 306.2 / 251.6 MiB per rank (#733). The strict boundaries add the FP32 projection rows; report the H100 numbers as they come.
- **Suites:** nothing fails only at #733 against #732 (list every failure on each side; on B200 both had the one known `test_h3_benchmark_timing.py::...[gather]` CPU failure).

B200 evidence for comparison: #732 [`pr732/reports/experiments/h3-sharded-model`](https://github.com/fusheng-ji/RL-Kernel/tree/f8d191cbd0a315aba1251a57f68ff6a0fbc4b0d3/pr732/reports/experiments/h3-sharded-model), #733 [`pr733/reports/experiments/h3-parallel-lora`](https://github.com/fusheng-ji/RL-Kernel/tree/4853e0d9ce55156acf8fde0b97335da4976248be/pr733/reports/experiments/h3-parallel-lora).

---

## S2. #735, strict boundaries (1 x 4 H100, exclusive, `<E2>`)

### Environment

`<E2>` is the final run's stack: torch 2.14.0+cu130, diffusers 0.43.0, vLLM 0.33.0, vLLM-Omni `6039984e397d3b4b28afd4538fa0207594e3ecba`, verl `2b4ed682`, verl-omni `a634f4d8` (editable at `<VERL_OMNI>`). `pip-list-e2-b200.txt` here is the B200 environment these runs are compared with; match it (`vllm`, `vllm-omni`, `verl`, `verl-omni`, `diffusers`, `transformers`, `triton`, `torch*`). If `<E2>` from the PR A round lacks vLLM / vLLM-Omni, add exactly those two (`vllm==0.33.0`, `vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@6039984e397d3b4b28afd4538fa0207594e3ecba`) without changing torch, and record the before / after `pip list`.

Use a short node-local `TMPDIR` (vLLM's IPC socket paths fail when it is long), `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `HF_HUB_OFFLINE=1`.

### Code

RL-Kernel builds on torch 2.14 only with the C++20 fix of #734. `h3-hopper-strict-20261122.bundle` holds one commit, `5f1a750b`: #735's head `4a4fe0c0` merged with #734's head `dcc8535a` (only `build_tools/extensions.py` and its test change):

```bash
cd <REPO>
git fetch https://github.com/RL-Align/RL-Kernel.git pull/735/head pull/734/head   # the bundle's prerequisites
git bundle verify <THIS_DIR>/h3-hopper-strict-20261122.bundle
git fetch <THIS_DIR>/h3-hopper-strict-20261122.bundle refs/heads/h100/h3-735-with-734:refs/heads/h100/h3-735-with-734
git worktree add --detach <S735> h100/h3-735-with-734   # 5f1a750b
cd <S735> && RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 <E2>/bin/python setup.py build_ext --inplace
```

### Runs

The G3 tool, once with LoRA B random and once with B = 0, each run with all three stages (the engine on GPUs 0-3; the canonical replay on GPU 0 after the engine has exited; the actor check on the CPU):

```bash
cd <S735>; export PYTHONPATH=<S735>
T=tools/validation/models/h3_omni_rollout_report.py
for b in random zero; do
  O=<OUT>/S2/lora-$b; mkdir -p $O; extra=""; [ $b = zero ] && extra="--lora-b-zero"
  CUDA_VISIBLE_DEVICES=0,1,2,3 <E2>/bin/python $T engine --checkpoint <CKPT> --engine-model <CKPT>/FL2VA --out $O \
    --tp 4 --max-num-seqs 2 --release-weights --strict-boundaries --lora-rank 64 $extra > $O/engine.log 2>&1
  CUDA_VISIBLE_DEVICES=0 <E2>/bin/python $T canonical --checkpoint <CKPT> --out $O > $O/canonical.log 2>&1
  CUDA_VISIBLE_DEVICES="" <E2>/bin/python $T actor --checkpoint <CKPT> --out $O > $O/actor.log 2>&1
done
CUDA_VISIBLE_DEVICES="" <E2>/bin/python -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3/test_h3_omni_model.py > <OUT>/S2/omni-cpu.txt 2>&1
CUDA_VISIBLE_DEVICES=0 RL_KERNEL_H3_WEIGHTS=<CKPT> <E2>/bin/python -m pytest -q -p no:cacheprovider -rs tests/models/minimax_h3/test_h3_omni_model.py > <OUT>/S2/omni-gpu.txt 2>&1
```

(`RL_KERNEL_H3_WEIGHTS` must be the directory that holds `transformer/`; with a wrong path the GPU tests skip instead of failing, so check they ran.)

### S2 expected (B200, provider boundaries, for both B random and B = 0)

- `engine.json` `pass: true`, with `install[*].strict_boundaries: true` on all 4 ranks:
  - `ranks_agree` (4 ranks x 8 DiT calls), `batched_calls_with_an_alone_twin` 4 and `batched_calls_without_an_alone_twin` 0;
  - 2 `captures`, each `in_run: true`, and no `record_errors`;
  - `lora.pass`: the sync with one changed element on rank 2 (block 17 `attn.to_v.A[3, 5]`) refused on every rank, the decoy kept, then the senders' digests on 650 shards per rank;
  - per rank `resident_param_bytes` 37,379,054,144 and `engine_parameter_bytes_released` 16,570,990,400 on B200 (report the H100 values).
- `canonical.json` `pass: true`: both captures replayed on one GPU, video and audio bitwise, `in_rollout_digests` true.
- `actor.json` `pass: true`: 3957 rows per call, `differ` empty.
- `omni-cpu.txt`: 38 passed, 3 skipped (the GPU tests); `omni-gpu.txt`: every test ran and passed.
- Timings (`engine.json` `seconds`): B200 A alone about 9 s, B alone 3.9 s, A + B 6.8 s; report the H100 ones.

B200 evidence for comparison: [`pr735/reports/experiments/h3-omni-rollout`](https://github.com/fusheng-ji/RL-Kernel/tree/e09678c28f389f6f10d307b595138ce33a72bff3/pr735/reports/experiments/h3-omni-rollout).

---

## What to send back

One fork branch, `handoff/h100-h3-hopper-strict-results-20261122`, cut from upstream `test-h3` (`12e81c36`), with `results/S1/` and `results/S2/` (each run's folder as written, plus `commit.txt`, `nvidia-smi.txt`, the environment line `python -c "import torch, sys; print(torch.__version__, torch.version.cuda, sys.version)"`), `pip-list-filtered.txt` for `<E1>` and `<E2>`, and `SUMMARY.md`: per run the verdict and every deviation from the expectation above, the memory per rank, and the S2 timings. The two tasks are independent: push each as soon as it is done (S2 may arrive first). Redaction scan before every push.
