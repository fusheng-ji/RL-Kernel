# H100 handoff: #557 EP4 chain on 4 x H100 (2026-10-29)

**For:** the agent on the Hopper (H100) machine. **Owner of every decision outside this file:** Wenbo. Stop and ask when something is not covered.
**The rules of `HANDOFF.md` (v2) apply unchanged**: section 2 "may / may not", section 5 desensitization, `evidence_check.sh`, one UTC clock, report format. In short: no branch pushes, no force, no PR/comment edits, no raw logs or denylist in git, nothing pushed to `pr_assets` until Wenbo says "push evidence". Do not edit or delete any existing `hopper-qwen3-next/` directory on `pr_assets`; the new data is a separate, labelled set.

## 1. Why

@Flink-ddd's change request on #557 (2026-10-19) asked for the EP4 forward/backward and rollout/replay parity checks on Hopper. Tests and gates already ran on H100 at `a705d287`. The full-model EP4 chain (`preflight`, `layers`, `l2`) was run earlier with **local changes not in the PR**; that result was withdrawn (`pr_assets` commit `927b93d6`). Goal: run the chain on the public head with **no local patch**, or find out exactly why it cannot run.

## 2. What to test (pinned)

| PR | fork branch | pinned head |
|---|---|---|
| #557 | `feat/ws1-qwen3-next-distributed-logprob-ep` | `88c4f02e` (verified on the fork 2026-10-29) |

- Head check is mandatory: `git rev-parse --short=8 HEAD` in a **fresh worktree** must equal `88c4f02e`. On mismatch stop and report.
- **No local patch of any kind.** `hopper_preflight_2a8f18e4.patch` is NOT needed: the Hopper allowance is already in this head (`a705d287`). If something fails, do not patch; report.
- Fresh venv and fresh build for this run. Do not reuse older worktrees or their `_C`. Verify: `python -c "import rl_engine._C as c; print(c.__file__)"` points into the worktree under test.

## 3. Environment (record in `env.txt`)

`date -u +%FT%TZ`; `nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader`; exactly 4 H100 visible (`H100`, cc 9.0); torch / CUDA / vLLM / NCCL / Megatron-core / TE versions; CUDA toolkit 13.0 (`nvcc --version`); `ninja` on PATH; `TORCH_CUDA_ARCH_LIST=9.0`; head SHA; `_C.__file__`; `pip freeze` (desensitized). Expected: torch 2.13.0+cu130, vLLM 0.30.0, NCCL 2.29.7.

## 4. Run protocol

| step | command | GPUs |
|---|---|---|
| 0. build | `python setup.py build_ext --inplace` | none needed |
| 1. unit | `python -m pytest -q -p no:cacheprovider tests/models/qwen3_next/test_qwen3_next_ep.py tests/models/qwen3_next/test_qwen3_next_chain_report.py` | 4 |
| 2. EP4 MoE gate | `TRITON_F32_DEFAULT=ieee torchrun --nproc-per-node 4 tools/validation/models/qwen3_next/qwen3_next_ep_moe_check.py` plus the extra arguments in the "Test" section of the #557 PR body (copy them verbatim) | 4 |
| 3. chain | `python tools/validation/models/qwen3_next/qwen3_next_run.py --stage all --until l2 --ep 4 --runs-root <runs> --checkpoint <ckpt> --cuda-home <cuda-13.0> --python <venv>/bin/python` | 4 |

- Copy the exact checkpoint revision from the #557 PR body (the pinned `Qwen3-Next-80B-A3B-Instruct` revision). A different revision is a different claim.
- Expected result (B200 reference): `preflight` passed, `layers` (with `moe_ep`) passed, `l2` **16 / 16 raw-bit**. The H100 result must be stated as measured, whatever it is. B200 numbers are never Hopper evidence.
- Stop at `--until l2`. EP4 training stays fail-closed; do not run training or export stages.

## 5. Known risks: report, do not work around

0. **Likely cause of the withdrawn run (found 2026-10-29).** The #543/#544 H100 results say full-model `l2` on 80 GB cards needed local commits that load the Megatron recompute **without DDP / optimizer** (no giant FP32 grad buffer); the patch is `pr_assets/hopper-qwen3-next/pr503/hopper_recompute_no_ddp.patch`. Expect the unpatched `l2` to hit OOM at the recompute load. Do NOT apply that patch in this run. Record where it fails (stage, rank, `nvidia-smi` memory) and stop after `preflight` and `layers`; Wenbo decides whether the no-DDP loading becomes a public commit. (Wenbo has written it as a local, unpushed commit on #543: `HANDOFF-543-544-CHAIN.md` and `543-hopper-patches/`. #557's own head does not contain it yet, so this run stays unpatched.)
1. **Memory.** 80B BF16 at TP4 is about 40 GB per GPU for weights; vLLM and the VIME recompute may not both fit in 80 GB. If you hit OOM: record `nvidia-smi` memory at the failing step, which stage and which engine, and stop. Do not change batch size, sequence length, `gpu_memory_utilization` or any code to make it fit; those change the claim. Report them as a candidate, Wenbo decides.
2. **Why the earlier run needed local changes is unknown.** If a step fails, report the failing command, the full error (desensitized) and your diagnosis. Do not patch.
3. **FlashInfer** `fused_moe_90` once failed to link on this machine (`cannot find -lnvrtc`). If the chain touches it, record the error verbatim and mark "not available"; do not add a symlink without asking, and list it if you do.
4. Run each step once on a clean tree. If a run is repeated, keep both and say which is which. No cherry-picking.

## 6. Report (one file, fixed format)

`report-557-ep4-chain.md` with: env table; head check output; per step: command (first line), start/end UTC, exit code, result counts; for the chain: stage pass/fail and the L2 case count with raw-bit equal count; list of any deviation (none expected); list of local patches (must be "none"). Then:
- Desensitize per `HANDOFF.md` section 5 and run `evidence_check.sh <results dir> <private denylist>`; it must exit 0.
- Raw logs go to the private archive only.
- Deliver the desensitized set under `pending/h100-qwen3-next-ep4/pr557/` with `SHA256SUMS`. **Do not push.** Wait for Wenbo's "push evidence" (target: `pr_assets/hopper-qwen3-next/pr557/chain/`, one fast-forward commit).

## 7. Out of scope

#543/#544 chain on H100 (same local-patch problem; separate round after this one), #503/#507/#549 head re-verification, EP training, other EP sizes, H200.
