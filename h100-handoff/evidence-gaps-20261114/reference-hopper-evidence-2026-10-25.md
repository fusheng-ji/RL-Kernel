# Hopper (H100) evidence handoff: PRs outside the RFC #420 / #428 stacks

Rewritten 2026-10-25. Only H100 work is listed; whatever B200 already covers is dropped. Local note with paths: redact before anything goes into a PR.
B200 reference results for the same heads: `pr<N>/b200-gaps/results/` on the local branch `local/b200-gap-evidence` (based on `pr_assets`); compare against them, never overwrite them. Runners are in `pr<N>/b200-gaps/run.sh` on that branch.

## Rules for every PR
1. One PR, one worktree, detached at the exact head SHA below (re-check `gh pr view N --json headRefOid`), `git status --short` empty before and after. Never edit a runner to make a number pass; commit runner changes first.
2. Build with `TORCH_CUDA_ARCH_LIST=9.0` (ninja on PATH, `MAX_JOBS=32`). Run from the worktree and print `rl_engine.kernels.ops.base._C.__file__`: it must be inside the worktree.
3. Put an environment header at the top of every result: GPU name, driver, CUDA, torch, triton, python, head SHA, `git status`, exact command lines.
4. Report passed / failed / skipped separately with skip reasons (`-rs`). Failures are results; do not retry until green. Do not reuse B200 counts.
5. Redact before commit: cluster, partition, job and node names, hostnames, absolute paths, usernames, personal email, IPs, GPU UUID / PCI bus id. Replace with neutral labels (`H100-80GB`, `<repo>/...`).
6. Evidence goes to `pr<N>/hopper/` on `pr_assets`; commits `git commit -s`, no AI attribution lines. Stop at local commits plus a drafted "H100" paragraph for the PR-body Results (links fixed to the evidence commit SHA). Do not push or comment.
7. Do not mix with RFC #420 / #428 PRs.

## Per-PR tasks

### #576 h3-flash-attention-v1: `feat/h3-flash-attention-v1` @ `9f4d0756fdaf1d20e446760948f66bce7ad43be1` (base `test-h3`)
```bash
RL_KERNEL_H3_STRICT=1 RL_KERNEL_REQUIRE_EXT=1 TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=0 python -m pytest -p no:cacheprovider -q -rs tests/backends/cuda/test_h3_flash_attention.py tests/backends/cuda/test_h3_full_attention.py
for i in 1 2 3; do CUDA_VISIBLE_DEVICES=0 python tools/validation/models/h3_flash_attention_bench.py --out hopper-run$i.json; done
```
- Without the strict build flag the #553 tests skip and the bench has no #553 column.
- The PR body and doc table (H100, written at an older head) predate the FP32-output delta fix. Re-measure, refresh the table and the "1.6x to 2.3x slower than SDPA" range (the 12-token row is about 1.3x).
- Needs about 4.3 GB free for the 2**31-element rejection test.
- Differs from B200: the `mma.sync` path means flash bytes on H100 need not equal B200 bytes; claim only within-H100 repeat / batch / head-shard invariance.

### #495 fused logp backward: `fix/174-chunked-logp-backward` @ `e612412c815d02d4d05dac1c5fe44d4688dcc82c`
```bash
TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -p no:cacheprovider -q -rs \
 tests/backends/cuda/test_fused_logp_chunked_backward.py tests/backends/cuda/test_fused_logp_backward_indexing.py \
 tests/ops/logprob/test_logp.py tests/ops/logprob/test_deterministic_logp.py tests/ops/logprob/test_batch_invariant_logp.py \
 tests/ops/test_op_accuracy.py tests/e2e/test_rl_kernel_loss_step.py tests/validation/common/test_logprob_comparison.py \
 tests/validation/operators/test_gradient_invariance.py tests/validation/operators/test_ws1_gtest_gpu.py \
 tests/validation/models/test_ws1_chain_integration.py tests/contracts/test_tolerance_contract.py \
 tests/benchmarks/test_fused_logp_backward_evidence.py tests/benchmarks/test_fused_logp_provider_evidence.py
python benchmarks/fused_logp_backward_evidence.py --flash-attn-src <flash-attention @ 94e22c9, plus SOURCE.json {"commit": "94e22c9"}> --out report.json
```
- The runner needs `liger-kernel==0.8.4` and `fla-core==0.5.2`: unpack the wheels into a separate directory on `PYTHONPATH`; do not pip install into a shared env.
- Differs from B200: the SM90 tests (`fused_logp_sm90`, Hopper-only logp comparisons) newly run; report them separately. The 80 GB limit may skip the large-N memory test: say so. Two GPUs make the multi-GPU device-guard test run. On B200 the fused backward is 28.4 ms backward-only at 32768 rows (the PR body's 12.5 ms predates the 64-bit loop fix): compare H100 against the B200 report, not the body. `report.json` gives the H100 timing / peak-memory table (32768 rows x 151936) and the shared-gradient accuracy.

### #491 deterministic all-gather copy: `fix/deterministic-collective-fast-copy` @ `cc27cf5cb565857abc1d78ae77d1541872e820d1`
```bash
TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
python -m pytest -p no:cacheprovider -v -rs tests/distributed/test_deterministic_all_gather_sizes.py tests/distributed/collectives/test_deterministic_all_gather.py tests/distributed/collectives/test_transport_deterministic_collective.py
python -m pytest -p no:cacheprovider -v -rs tests/distributed/collectives/test_deterministic_all_reduce.py tests/distributed/collectives/test_deterministic_reduce_scatter.py
torchrun --standalone --nproc-per-node=<2|4|8> benchmarks/benchmark_deterministic_collectives.py --output after_w<N>.json
```
- "Before" = merge base `5ba769f681b7fa765ea6b29f9939454a858390ba`, built the same way, run with the head's benchmark script and `PYTHONPATH` set to the base worktree, back to back on the same GPUs.
- The existing H100 section covers 2 and 4 GPUs at `55cb531a`; missing is an 8-GPU run at this head (8-GPU tests skip below 8 GPUs). Check whether `kAllGatherSingleBlockMaxBytes = 64 KiB` still holds on H100 and say so if not; do not retune in this PR. State NVLink / NVSwitch vs PCIe topology in neutral words. On B200 the full 8-rank benchmark aborts in the fused `all_reduce` ("too many resources requested for launch") on both trees, so use `--operations all_gather all_gather_many` if H100 does the same; the reduce-scatter cross-TP test fails with an `outs=` TypeError (known on main): run the merge base as the baseline.

### #537 Qwen-Image adaln_mod_projection: `feat/qwenimage-adaln-mod-projection` @ `a73c66c5e2db7ec844e30ac763ce85d410179259` (base `test-qwenimage`)
```bash
KERNEL_ALIGN_USE_FAST_MATH=0 TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m pytest -p no:cacheprovider -q -rs tests/models/qwenimage tests/runtime/test_kernel_registry.py tests/contracts/test_tolerance_contract.py tests/build/test_build_platform_collectives.py tests/validation/operators/test_ws1_gtest_gpu.py tests/runtime/test_layout_compatibility.py
```
Then, one GPU, with `MODEL=Qwen/Qwen-Image @ 75e0b4be04f60ec59a75f475837eced720f823b6`: the acceptance chain in `pr537/b200-gaps/run.sh` (`adaln_evidence.py --repeats 10 --fp64-check`, `adaln_prior_art.py`, `adaln_reuse_bi_audit.py --mode fixed` for seeds 0/127/2026 x phases accuracy/accuracy_large and phases bi/workload/parameters, `adaln_parameter_screen.py`, `validate_adaln_repair.py`). Two GPUs are needed for the device-isolation test.
- Differs from B200: the BF16 path uses `mma.sync`, so H100 bytes need not equal B200 bytes. Claim only within-H100 repeat / batch invariance, no cross-architecture equality. The PR scope says "no H100 run": this adds it.

### #538 Qwen-Image final_adaln_out: `feat/qwenimage-final-adaln-out` @ `2d9680d7b8b9c0f7d150bd7cb296cf496652963a`
Same commands and rules as #537 (stacked on it; run #537 first; script `pr538/b200-gaps/run.sh`). Keep the long-sequence cases S = 4096 / 6889 / 6032. Optional: the Liger and flash-attention LayerNorm reuse modes (`vllm_liger`, `vllm_flashattn`) of `adaln_reuse_bi_audit.py`, which need those libraries in a separate `--target` directory.

### #536 HAL persistent autotuning: `feat/issue-50-hal-autotuning` @ `54307e1b175943e91ad5f20379003ce73c4dd071` (merge base `16b81175874ecf11d5ff05c00de3b69e9457591b`)
```bash
TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
export RL_KERNEL_MODE=auto RL_KERNEL_AUTOTUNE_CACHE_DIR=<private dir>
RL_KERNEL_AUTOTUNE={0|1} python -m pytest -p no:cacheprovider -q -rs tests/ops/gemm/test_det_gemm.py
python -m pytest -p no:cacheprovider -q -rs tests/platforms/test_persistent_autotuner.py tests/runtime/test_kernel_registry.py tests/ops/test_autotuning_gpu.py tests/ops/test_autotuning_candidates.py tests/benchmarks/test_autotuning_benchmark.py
RL_KERNEL_AUTOTUNE={0|1} python -m pytest -p no:cacheprovider -q -rs tests/ops     # also on the merge base, same build
python -m benchmarks.tuning.benchmark_autotuning --operator all --dtype bfloat16 --preset all --phase all --trials 5 --output issue50-performance.json
```
- Differs from B200: the SM90 deterministic GEMM cases may run or behave differently; report a diff against the merge base on the same box, not absolute counts. The cache key contains the device UUID and PCI bus id: redact them from cache dumps and logs. Keep the CUDA-graph cases. Four GPUs make the TP=4 case run.

### #547 recursive deterministic LogP autograd: `feat/b200-deterministic-logp-autograd` @ `ff1ad6350050b3b11bec86364f5c1bfaecf23f78`
```bash
KERNEL_ALIGN_FORCE_SM90=1 TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
python -m pytest -p no:cacheprovider -q -rs tests/ops/logprob/test_deterministic_logp_recursive.py
RL_KERNEL_LOGP_MAX_ORDER=10 python -m pytest -p no:cacheprovider -q -rs tests/ops/logprob/test_deterministic_logp_recursive.py -k 'recursive_orders and dtype3'
python -m pytest -p no:cacheprovider -q -rs tests/ops/logprob tests/runtime/test_layout_compatibility.py
python tools/validation/operators/probe_deterministic_logp_autograd.py --output recursive-backends.json
```
- Differs from B200: about 15 SM90-only tests (`batch_invariant_logp_sm90`, fused linear logp SM90) skip on B200 and need the `KERNEL_ALIGN_FORCE_SM90=1` build to run; report them separately. Keep the branch name; add an H100 section. Do not cite the local `run-validation.sh` workspace.

### #548 PPO Qwen3 adapters: `feat/issue-44-tp-ppo` @ `4585cd3183cf85864adc50f117c55fa92324f1e2`
Not a quick re-run. The Qwen3-8B candidate stage has not run on any hardware (raw reference and pinned environments are gone; see `pr548/b200-gaps/CANDIDATE-STAGE-PLAN.md`), so there is nothing to compare an H100 run against. Optional only: the unit tests (`tests/integrations/common/test_ppo.py`, `test_tp_scalar.py`; `tests/integrations/engines/train/test_ppo_adapters.py`, `test_ppo_bf16.py`, `test_qwen3_ppo_checkpoint.py`, `test_qwen3_vllm_install.py`) at this head, labelled "H100 smoke". The existing H100 section was measured at `17d19247`. Ask before doing more.

## Hand back per PR
`pr<N>/hopper/` with redacted raw logs, JSON / CSV, environment header, exact commands, head SHA, `git status`; a passed / failed / skipped table with skip reasons and a diff against the B200 results; the drafted H100 paragraph; anything that failed, unedited.
