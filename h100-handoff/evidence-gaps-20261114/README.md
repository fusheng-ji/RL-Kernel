# H100 handoff: evidence gaps on the open PRs (2026-11-14)

B200 work is shelved. Local note with paths: redact before anything goes into a PR. Recipes that are not repeated here are in `reference-hopper-evidence-2026-10-25.md` (this directory: rules and per-PR commands; its heads are outdated, so take only the commands) and in the PR descriptions' "Results (H100)" sections; the earlier raw runs are on the fork's `pr_assets` branch under `hopper-qwen3-next/pr<N>/`, `hopper-main/pr<N>/`, `pr495/hopper/`.

## What each PR's H100 evidence is at, versus the head to test now
Re-check every head with `gh pr view N --json headRefOid` before starting; a head that moved changes the target.

| PR | Current head | H100 evidence is at | Priority |
|---|---|---|---|
| #503 | `d5253a5c` | validation commit `f35f98cb` (a Hopper-only fix not in the PR) | **1: decides a code change** |
| #543 | `4e2c33af` | `4460db96` + `2a8f18e4` | 2 |
| #544 | `81be0811` | `91b77274` + `2a8f18e4` | 2 |
| #549 | `30c0e5a7` | `8e88cd82` | 2 |
| #557 | `f935bf3a` | `a705d287` (description: "not rerun at the head") | 2 |
| #551 | `2c699e3f` | comparison runner `58aa50c7` (same branch) | 2 |
| #507 | `1fef29b0` | `0fdbf80b` | 3 (owned by the #428 coordinator: run it, do not edit the PR) |
| #495 | `bc742c41` | full suite `a320e65a`; performance `b1906159` | 3 |
| #491 | `5882864c` | `55cb531a` | 3 |
| #536 | `4a83e4f8` | `f203661c` | 3 |
| #538 | `a24dc5fc` | `e23c4a97` | 4 |
| #547 | `eb465fd8` | `6b8b05e5` | 4 |
| #548 | - | skip: Flink's two threads need code changes first | - |

## Rules for every run
1. One PR, one detached worktree at the exact head SHA; `git status --short` empty before and after; never edit files in it. Hopper-only patches (below) are applied as a separate, named step and reported as such.
2. Build with `TORCH_CUDA_ARCH_LIST=9.0`, ninja on PATH, `MAX_JOBS=32`; print `rl_engine._C.__file__` and confirm it is inside the worktree. For SM90 paths add `KERNEL_ALIGN_FORCE_SM90=1` (and `KERNEL_ALIGN_DET_GEMM_SM90=1` for the Qwen3-Next chain), as in the existing H100 runs.
3. Header at the top of every result: GPU name and count, driver, CUDA, torch, triton, python, head SHA, `git status`, exact commands. Say if another job was running on the node.
4. Passed / failed / skipped are reported separately, with skip reasons (`-rs`). A failure is a result: do not retry until green, do not delete it.
5. Do not install into `envs/rlkernel`. Use the prior-art venv (Liger 0.8.4, FLA 0.5.2) or a separate directory on `PYTHONPATH` for extra libraries.
6. Redact before committing: cluster, partition, job and node names, hostnames, absolute paths, usernames, personal email, IPs, GPU UUID and PCI bus id (also in cache dumps and `command` / `build_note` fields in JSON).
7. Evidence goes to `pr_assets` under the existing per-PR directory (`pr<N>/hopper/` or `hopper-qwen3-next/pr<N>/`), named with the head SHA; commit `-s`, no AI attribution lines. Do not push PR branches and do not edit PR descriptions: hand back a drafted paragraph per PR.

## Priority 1: #503, does the PR head fail batch invariance on H100?
The description's H100 table was measured at `f35f98cb`, which replaces `torch.sum` in the `combine_routes` backward with a fixed-order row sum (`_CombineRoutes`). The PR head does not have that change. The description says the stock reduction makes `dx` non-invariant from batch 16 upward on H100.

Run on 4 x H100 at `d5253a5c`, with the Qwen3-Next build flags above:
1. **Without any patch**: the same batch-invariance checks that produced the H100 section (read the commands from `hopper-qwen3-next/pr503/` `info.txt` and logs), at minimum `tests/models/qwen3_next/test_*.py` and `check_*.py` and the TP4 MoE gate. Report which cases fail and at which batch sizes. This is the result I need most.
2. **With the patch** `hopper-qwen3-next/pr503/hopper_combine_routes_bi.patch` applied on top: same checks.
3. Do not apply `hopper_recompute_no_ddp.patch` here; it is for the full-model `l2` runs below.

Hand back both, side by side. Whether the fixed-order reduction goes into #503 is decided afterwards.

## Priority 2: the Qwen3-Next stack
The five stack PRs merge their predecessor, so the top of the stack (#551 `2c699e3f`, which contains #543, #544, #549, #557) is one build for all of them. Since the last H100 runs, the heads gained: a provider workflow with new package-version checks and a provider test (`test_qwen3_next_vllm_attention_metadata.py`), a gate script that exits 2 on malformed reports, router / TP guards, and merges of `main`. CI skips the `provider` job on forks, so none of that has run on a GPU runner.

### A. Provider job at the top of the stack (4 x H100, one build)
Repeat the steps of `.github/workflows/qwen3-next-provider-gpu.yml` at head `2c699e3f`, in order, as the job does:
```bash
# 1. audited runtime: torch 2.13.0+cu130, vllm 0.30.0, transformers 5.17.0, triton 3.7.1, flashinfer-python 0.6.18.post1
# 2. build
find . -name '_C*.so' -not -path './.git/*' -delete; rm -rf build
KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py build_ext --inplace
python3 tools/checks/ci_smoke.py
# 3. provider checks in their own process; any skip is a failure
TRITON_F32_DEFAULT=ieee python3 -m pytest -q -rs tests/models/qwen3_next/check_qwen3_next_norm_providers.py \
  tests/models/qwen3_next/check_gdn_recurrent_golden.py tests/models/qwen3_next/check_qwen3_next_forward.py \
  tests/models/qwen3_next/check_qwen3_next_attention.py tests/models/qwen3_next/check_qwen3_next_gdn_bridge.py \
  tests/models/qwen3_next/check_qwen3_next_gdn_sequence.py tests/models/qwen3_next/check_qwen3_next_conv_bridge.py \
  tests/models/qwen3_next/check_qwen3_next_core_matrix.py tests/models/qwen3_next/check_qwen3_next_shared_core.py \
  tests/models/qwen3_next/check_qwen3_next_paged_attention.py tests/models/qwen3_next/check_qwen3_next_batch_invariance.py \
  tests/models/qwen3_next/check_qwen3_next_bi_requests.py tests/models/qwen3_next/test_qwen3_next_vllm_attention_metadata.py
# 4. four GPUs: vLLM model contract and the EP4 real model
TRITON_F32_DEFAULT=ieee python3 -m pytest -q -rs tests/models/qwen3_next/check_qwen3_next_vllm_model_contract.py tests/models/qwen3_next/check_qwen3_next_vllm_real_model.py
```
If the workflow file at the head has more steps than listed, the file wins; list any step you skipped and why. `libnvrtc.so` needs to be on `LD_LIBRARY_PATH` (symlink to `libnvrtc.so.13`) for FlashInfer's sm90a JIT. At this head the four-GPU preflight accepts B200, H100 and H200 (from #557), so the `2a8f18e4` patch is not needed here.

### B. Chain stages at the top of the stack
`preflight`, `layers`, `l2` through `python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --until l2 ...` (arguments as in `docs/usage/qwen3-next-chain.md`), official checkpoint revision `9c7f2fbe`. Full-model `l2` on 4 x H100 needed `hopper_recompute_no_ddp.patch` last time (Megatron recompute without DDP / optimizer); apply it only for this step and say so. Then `qwen3_next_chain_report.py` and `bash ci/scripts/run_qwen3_next_chain_gate.sh chain-report.json` (a malformed report now exits 2, a failed gate exits 1; report the exit code).

### C. Per-PR claims that name a commit older than the head
Repeat the run recorded under `hopper-qwen3-next/pr<N>/` (commands are in its logs and `info.txt`), at that PR's own head, only for these:
- **#549** (`30c0e5a7`): TP4 parity over the 13 NCCL settings (`tools/validation/models/qwen3_next/qwen3_next_logprob_parity.py` + `_compare.py`, commands in the #549 description). The logprob prior-art is no longer on #549's head (moved to #551): see D.
- **#557** (`f935bf3a`): the 814-test gate run and the EP matrix (`ep_matrix/`). This answers Flink's Hopper change request at the current head; H200 stays untested.
- **#543** and **#544**: skip if A and B pass at the top of the stack, since the top contains them; say so. If A or B fail, rerun at their own heads with `hopper_preflight_2a8f18e4.patch`.
- **#551**: the comparison runner at its own head; keep the raw JSON. The B200 figures were re-rendered from the release ZIPs with the head's plot scripts (2026-11-14, pixel-identical to the published PNGs), so no plot work is needed for B200.

### D. Logprob prior-art on H100 under all 13 NCCL settings (at #551 `2c699e3f`)
**Why:** the H100 prior-art on `pr_assets` (`hopper-qwen3-next/pr549/logprob/report.json`, runner at `77d59a59`) ran under NCCL defaults only (`nccl_settings` has one entry), so its `nccl_setting_invariant: true` for all four candidates is untested. On B200 the same check over 13 settings found Megatron-core `vocab_parallel_cross_entropy` and VIME's default logprob give 2 results; the H100 report cannot show that. It also lives under `logprob/`, while the plot script reads `prior-art/report.json`.

The runner is `benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py`, only on #551's head (moved out of #549 by the layout review). Use the build from A (same worktree, same head). It needs the training environment for two candidates: Megatron-core 0.16.0rc0 and VIME 0.3.2 importable (the earlier H100 run had them; a separate `PYTHONPATH` directory, not `envs/rlkernel`). A candidate that cannot load is recorded as `unavailable` instead of failing: check that none of `rl_kernel_ws2`, `vllm_gather`, `megatron_ce`, `vime_default` is.

The 13 settings, each a separate 4-GPU launch, directory names as in the B200 ZIP:

| directory | env |
|---|---|
| `tp4-default` | none |
| `tp4-ring-{ll,ll128,simple}-c1` | `NCCL_ALGO=Ring NCCL_PROTO={LL,LL128,Simple} NCCL_MAX_NCHANNELS=1` |
| `tp4-ring-{ll,ll128,simple}-cdefault` | `NCCL_ALGO=Ring NCCL_PROTO={LL,LL128,Simple}` |
| `tp4-tree-{ll,ll128,simple}-c1` | `NCCL_ALGO=allreduce:tree NCCL_PROTO={LL,LL128,Simple} NCCL_MAX_NCHANNELS=1` |
| `tp4-tree-{ll,ll128,simple}-cdefault` | `NCCL_ALGO=allreduce:tree NCCL_PROTO={LL,LL128,Simple}` |

```bash
E=<out>/qwen3-next-distributed-logprob-h100
# 13 TP4 launches, for example:
torchrun --nproc-per-node 4 benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py \
  --checkpoint <ckpt> --output $E/prior-art/runs/tp4-default
NCCL_ALGO=Ring NCCL_PROTO=LL NCCL_MAX_NCHANNELS=1 torchrun --nproc-per-node 4 \
  benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py --checkpoint <ckpt> --output $E/prior-art/runs/tp4-ring-ll-c1
# ... the other 11
torchrun --nproc-per-node 1 benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py \
  --checkpoint <ckpt> --output $E/prior-art/runs/tp1
# merge: tp4-default first, tp1 last
python benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py \
  --merge $E/prior-art/runs/tp4-default $E/prior-art/runs/tp4-ring-* $E/prior-art/runs/tp4-tree-* $E/prior-art/runs/tp1 \
  --out $E/prior-art/report.json
```
Checkpoint: official `Qwen/Qwen3-Next-80B-A3B-Instruct` revision `9c7f2fbe`. Check `report.json`: `nccl_settings` has 13 entries and `tracked_tree_dirty` is false.

For the figure, put the #549 parity runs from C beside it as `$E/parity/runs/<setting>/rank-{0..3}.json` (with a `default` run) and `$E/parity/report.json`, then
```bash
python benchmarks/models/qwen3_next/plot_qwen3_next_distributed_logprob.py $E   # writes $E/figure.png, no GPU
```
If the parity runs in C are at `30c0e5a7` and the prior art at `2c699e3f`, say so in a README in `$E` (the B200 ZIP records two commits the same way). Optional: `--miles-src` / `--verl-src` with checkouts of `radixark/miles@23d3d38` and `volcengine/verl@e5a627c` add Miles and verl candidates; the plot does not show them, so only if time.

Archive under `pr_assets` `hopper-qwen3-next/pr551/distributed-logprob/` (redacted), next to the old `pr549/logprob/`, which stays as is. Hand back: per candidate, `nccl_setting_invariant`, TP4 == TP1, batch invariance, max diff vs FP64, TP4 latency, and the number of distinct `candidates.<key>.selected_sha256` over the 13 `rank-0.json` files (the merged report only has the boolean; count it from the runs). Say whether Megatron-core and VIME again give more than one result, and include the figure.

## Priority 3
### #507 (`1fef29b0`)
The D=256 attention gates and prior-art at the current head (the description's run is at `0fdbf80b`). Same build flags as A. Report only; do not edit.

### #495 (`bc742c41`)
Single node, two GPUs. Build `TORCH_CUDA_ARCH_LIST=9.0`, then a second build with `KERNEL_ALIGN_FORCE_SM90=1`.
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -p no:cacheprovider -q -rs \
 tests/backends/cuda/test_fused_logp_chunked_backward.py tests/backends/cuda/test_fused_logp_backward_indexing.py \
 tests/ops/logprob/test_logp.py tests/ops/logprob/test_deterministic_logp.py tests/ops/logprob/test_batch_invariant_logp.py \
 tests/ops/test_op_accuracy.py tests/e2e/test_rl_kernel_loss_step.py tests/validation/common/test_logprob_comparison.py \
 tests/validation/operators/test_gradient_invariance.py tests/validation/operators/test_ws1_gtest_gpu.py \
 tests/validation/models/test_ws1_chain_integration.py tests/contracts/test_tolerance_contract.py \
 tests/benchmarks/test_fused_logp_backward_evidence.py tests/benchmarks/test_fused_logp_provider_evidence.py
```
The fused kernel is unchanged since `61d30a0c`, so this repeats `a320e65a`; the 18 SM90 tests are reported separately. The 80 GB limit may skip the largest-N memory test: say so. The performance check at `b1906159` already exists (`pr495/hopper/`); do not repeat it. The B200 table regeneration is shelved.

### #491 (`5882864c`)
Benchmark and plot scripts are now under `benchmarks/distributed/`, their tests in `tests/benchmarks/`, the size test in `tests/distributed/collectives/`; the evidence at `55cb531a` used the old paths. "Before" = `upstream/main` (check its current SHA) built the same way, run with this head's benchmark script and `PYTHONPATH` pointing at the main worktree, back to back on the same GPUs.
```bash
TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
python -m pytest -p no:cacheprovider -v -rs tests/distributed/collectives/ tests/benchmarks/
torchrun --standalone --nproc-per-node=<2|4|8> benchmarks/distributed/benchmark_deterministic_collectives.py --output after_w<N>.json
```
2 and 4 GPUs at least; **8 GPUs if a node has them** (the description says there was no 8-GPU run). State the topology (NVLink / NVSwitch vs PCIe) in neutral words, and whether the 64 KiB single-block threshold still holds on H100; do not retune.

### #536 (`4a83e4f8`)
Build `TORCH_CUDA_ARCH_LIST=9.0`; private autotune cache directory (`RL_KERNEL_AUTOTUNE_CACHE_DIR`), `RL_KERNEL_MODE=auto`.
```bash
RL_KERNEL_AUTOTUNE={0|1} python -m pytest -p no:cacheprovider -q -rs tests/ops/gemm/test_det_gemm.py
python -m pytest -p no:cacheprovider -q -rs tests/platforms/test_persistent_autotuner.py tests/runtime/test_kernel_registry.py tests/ops/test_autotuning_gpu.py tests/ops/test_autotuning_candidates.py tests/benchmarks/test_autotuning_benchmark.py
RL_KERNEL_AUTOTUNE={0|1} python -m pytest -p no:cacheprovider -q -rs tests/ops      # also on upstream/main, same build, same box
```
Report a diff against `upstream/main` on the same box, not absolute counts. The head has a functional change since the description's run (`e5373328`, skip tuning when the profiling copies do not fit): keep or add a case that exercises it. Four GPUs make the TP=4 case run. Redact device UUID and PCI bus id in cache dumps. ROCm is not covered by this handoff.

## Priority 4 (quick, only if time)
- **#538** (`a24dc5fc`): `CUDA_VISIBLE_DEVICES=0,1 python -m pytest -p no:cacheprovider -q -rs tests/models/qwenimage` with `KERNEL_ALIGN_USE_FAST_MATH=0` in the build; the changed test is `test_final_scale_then_shift_and_layernorm`. Claim only within-H100 invariance (BF16 uses `mma.sync`).
- **#547** (`eb465fd8`): the SM90 commands of `reference-hopper-evidence-2026-10-25.md` (this directory; its heads are outdated, use only its commands) (`KERNEL_ALIGN_FORCE_SM90=1` build; `test_deterministic_logp_recursive.py`; `RL_KERNEL_LOGP_MAX_ORDER=10` case; `tests/ops/logprob`). The head only adds merges of `main`.

## Hand back
Per PR, in the directory named above: redacted raw logs and JSON, the header, exact commands, head SHA and `git status`; a passed / failed / skipped table with skip reasons; anything that failed, unedited; a drafted "H100" paragraph naming the head and any patch that was applied. For #503 also the two runs side by side. For the stack, state clearly which PRs the top-of-stack run covers.
