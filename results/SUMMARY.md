# Qwen3-Next stack on H100, 2026-11-18: SUMMARY

Handoff `handoff/h100-flink-round-20261116` @ `3e8a61bf` (`README.md` and `update-20261118/README.md`). Hardware: 4 x NVIDIA H100 per node (95830 MiB each), driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, vLLM 0.30.0, flashinfer-python 0.6.18.post1, python 3.12.15, NCCL 2.29.7. Topology: 4 GPUs per node, every pair NVLink-connected (NV6, `step0-.../nvidia-smi.txt`). Every run used a detached, clean worktree (`git status --porcelain --untracked-files=no` empty before and after, see each `header.txt` / `status-after.txt`), `TORCH_CUDA_ARCH_LIST=9.0`, and for Qwen3-Next `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1`; `rl_engine._C.__file__` was printed and is inside the worktree (`c-file.txt`). **No patch was applied in any run. `hopper_recompute_no_ddp.patch` was not needed: `l2` was not run.**

## What was run and what was not
| Step | Status |
|---|---|
| 0: `run_probe.sh`, six cases | **run; all six cases fail** (details below) |
| 0: real 80B smoke with `VLLM_ALLREDUCE_USE_FLASHINFER=0` | **not run**: the condition was "only if `q35-no-fi-ar` passes"; it did not |
| 3: EP4 MoE gate (#557) | run at **`7e312b26`** (the new #557 head, not `62487710`): **passed, 10 cases, all 4 ranks** |
| 1-3: chain `preflight`/`layers`/`l2` (#543), 97-cell `bi` (#544), EP4 chain (#557) | **not run**: they start vLLM, and step 0 failed. The handoff says not to work around a step 0 failure but to hand it back first. |
| 4: #503 `8e33c453`, 9 test files | run: **323 passed, 0 failed, 0 skipped** |
| 5: #507 attention prior art at `2c0cd1e1` | run; 7 of 10 candidates measured, 3 unavailable (finding below) |

## Heads
The PR heads moved while the work was in progress (pushes on 2026-11-18 between 21:51 and 21:57 UTC). Heads at the time of the runs, from `gh pr view N --json headRefOid`: #543 `5375881f`, #544 `c6bd8466`, #549 `14ef4bda`, #557 `7e312b26`, #551 `2c0cd1e1`, #503 `8e33c453`, #507 `1fef29b0`. The handoff README lists `85675a74`, `c49419ae`, `e8195782`, `62487710`, `ea999fd4`. I tested the new heads where something ran (#557 `7e312b26`, #551 `2c0cd1e1`, #503 `8e33c453`). `2c0cd1e1` is the SHA of the local bundle commit, so step 5 now runs on a pushed commit.

## Step 0: the abort reproduces on every case, and the environment differs from the B200 reference
Command: `run_probe.sh <python> <Qwen3.5-0.8B dir> <Qwen2-1.5B-Instruct dir> <out>`; cold FlashInfer JIT cache per launch; weights dummy; config and tokenizer only from the Hub. The launcher header (`step0-rollout-smoke-probe/header.txt`) lists every `libtvm_ffi` copy and the default FlashInfer cache.

| Case | rc | `__ffi_repr__` abort lines | FlashInfer workspace |
|---|---|---|---|
| `q35-auto` | 1 | 4 | `trtllm` |
| `q35-auto-nofuse` | 1 | 4 | `trtllm` |
| `q35-no-fi-ar` | 1 | 4 | none (all-reduce FlashInfer off) |
| `q35-gdntriton` | 1 | 4 | `trtllm` |
| `q2-auto` | 1 | 4 | `trtllm` |
| `q2-no-fi-ar` | 1 | 2 | none |

Every case dies with `tvm::ffi::Error: TypeAttr __ffi_repr__ is already registered for type index 132`; there are four rank processes, so four abort messages per case (two in the last one, which was run by itself, see below). Reading:
- A model without GDN (`q2-*`) aborts too, and so does the case that keeps FlashInfer's all-reduce out (`q35-no-fi-ar`, `q2-no-fi-ar`): the abort is not tied to GDN, to the fusion, or to the trtllm all-reduce workspace. In the logs it comes after the Triton warm-up and the first memory profiling line, after the workspace log line when there is one.
- **Environment difference.** The B200 reference used apache-tvm-ffi **0.1.11**, cuda-python 13.4.1 and python 3.12.14 (`b200-reference/versions.txt`). In this environment the interpreter picks `apache-tvm-ffi` **0.1.16**, cuda-python 13.4.4 (header), because a user-site copy (outside the virtualenv) shadows the packages of the audited environment. Nothing in the shared environment was changed.
- **Diagnostic, outside the handoff's six cases and not used for any result:** the same `q35-auto` case with `apache-tvm-ffi==0.1.11` unpacked into a separate directory and put first on `PYTHONPATH` (`second-launch-q2-no-fi-ar/diag-q35-auto-tvm0111.*`): **rc 0, no abort, 8 generated tokens**. So on this node the abort goes away with the B200 reference version of `apache-tvm-ffi`. This is one run of one case; it was not repeated and it is not an environment fix that was applied to anything else.
- Because of this I did **not** run the vLLM chain steps with that variant: the handoff says to hand step 0 back before working around it. Whether the chain should be rerun with `apache-tvm-ffi==0.1.11` on `PYTHONPATH` (or in a clean environment) is for you to decide; it is cheap to do.

Launch notes (kept as they happened):
- The first launch (`attempt0-stopped-after-q35-auto/`) was stopped by me after its first case because I had redacted the live output directory in place, which would have corrupted the logs of the running case. The `q35-auto` log of that launch (rc 1, 3 abort lines) is kept; the six-case launch was restarted from scratch.
- During the second launch the Slurm allocation that hosted it ended after five cases had finished (`sacct`: CANCELLED, 52 minutes after its start; I did not cancel it and do not know who or what did). The sixth case, `q2-no-fi-ar`, was rerun by itself on another node with the identical command and settings (`second-launch-q2-no-fi-ar/`).

## Step 3: EP4 MoE gate, #557 `7e312b26`
`python -m torch.distributed.run --nproc-per-node 4 tools/validation/models/qwen3_next/qwen3_next_ep_moe_check.py --checkpoint <Qwen3-Next-80B-A3B-Instruct, revision 9c7f2fbe> --output <dir>`, layer-0 weights, no vLLM. Result: `status: passed` on ranks 0-3, 10 cases per rank (`hf-load-export-each-expert-once`, `ep4-vs-ep1-router-combine-backward-{8,64,256,1024}`, `ep4-vs-ep1-chunk-reorder-train-{8,64,256,1024}`, `empty-rank-participation`), `rl_kernel_commit` `7e312b26`, `tracked_tree_dirty: false`; B200 reference at `62487710`: 10 cases passed. The first launch (`attempt1-wrong-torchrun-gate.log`) failed before any test: the `torchrun` on `PATH` belongs to another Python (torch 2.8) and raised `PackageNotFoundError: vllm`. It is a launcher error, kept unedited; the second launch used `python -m torch.distributed.run` of the audited interpreter. (The exit code of the first launch is wrongly 0 in `steps.txt`: the logging command substituted `date` before reading `$?`; the log shows the failure.)

## Step 4: #503 `8e33c453` (no vLLM, SM90 flags)
Nine files as in the B200 run (`hopper-qwen3-next/pr503/b200-8e33c453/steps.txt`) plus its `bias_check.py`: **323 passed, 0 failed, 0 skipped**, the same count as B200; `bias_check` rc 0 (`bias.grad` is `(64,)` of zeros for `requires_grad=True`, `None` otherwise). `RESULTS.md` in the folder.

## Step 5: #507 attention prior art at `2c0cd1e1` (timing fix), 1 H100
Commands as in `update-20261118/README.md` item B (build with the SM90 flags; `te_fused` in its own process; `--merge`; plot). `report.json`: `rl_kernel_commit` `2c0cd1e12668c294e16c55d539af87fb29d2ac64`, `tracked_tree_dirty: false`.

Environment differs from the B200 rerun: **Transformer Engine 2.18.0 (not 2.20.2), megatron-core 0.16.0rc0 at Megatron-LM `1dcf0daf` (not `52fbcbc18` / 0.20.0)**. In this environment, `te_training`, `megatron_local` and `te_fused` are all reported `unavailable: RuntimeError: cudnn shared object not found.` (a finding, not fixed: nothing was installed and no library path was changed). So 7 candidates were measured; the TE and Megatron-local rows are missing.

| Check | Expected | Observed (7 measured candidates) |
|---|---|---|
| batch invariant yes | RL-Kernel, FA-2 split=1, Triton 2D, FlashInfer | yes for all four |
| batch invariant no | SDPA, FA-2 auto, FA-4 | no for all three |
| TE training, TE fused, Megatron local | no / yes / yes | unavailable |
| max abs error vs FP64 | about 6.7e-3 | 6.68e-3 (all seven) |
| prefill latency, FlashInfer (512 / 2048 / 8192) | below 364 / 420 / 793 us | 76 / 114 / 351 us |
| prefill latency, Triton 2D (512 / 2048 / 8192) | below 412 / 527 / 1949 us | 151 / 278 / 1622 us |

Files: `report.json`, `report-main.json`, `figure.png`, `attn-main.log`, `attn-te.log`, `plot.log`, `versions.log`.

## Drafted "H100" paragraphs
- **#557** (`7e312b26`): 4 x H100 (driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, vLLM 0.30.0, NCCL 2.29.7), official checkpoint `9c7f2fbe`, SM90 flags, detached clean worktree, no patch: the EP4 MoE gate passes on all four ranks, 10 cases, including `ep4-vs-ep1-router-combine-backward-{8,64,256,1024}`. The chain (`preflight`, `layers`, `l2`) was not run at this head: the vLLM rollout smoke of `preflight` aborts on this H100 setup (see #543 below).
- **#543 / #544 / #551** (heads `5375881f`, `c6bd8466`, `2c0cd1e1`): chain not run on H100 in this round. The `rollout-smoke` abort (`__ffi_repr__ is already registered`) reproduces in the six probe cases with `apache-tvm-ffi` 0.1.16 in the environment, and not in one diagnostic run with the B200-reference `apache-tvm-ffi` 0.1.11. Chain results at these heads are pending a decision on the environment.
- **#503** (`8e33c453`): 4 x H100 node, SM90 flags, no patch: the nine files of the B200 run give 323 passed / 0 failed / 0 skipped, and the empty-input bias-gradient check passes.
- **#507** (`2c0cd1e1`, prior art): see step 5; 7 of 10 candidates measured, with the invariance verdicts of the current H100 table and lower FlashInfer and Triton 2D latencies; TE and Megatron-local candidates unavailable in this environment (`cudnn shared object not found`), with TE 2.18.0 and megatron-core 0.16.0rc0.

`nvidia-smi.txt` in each step folder: `nvidia-smi --query-gpu=name,driver_version --format=csv` and `nvidia-smi topo -m` (the topology of the node that ran step 0 is in `step0-rollout-smoke-probe/header.txt`; all nodes have the same 4 x H100 NV6 layout).
