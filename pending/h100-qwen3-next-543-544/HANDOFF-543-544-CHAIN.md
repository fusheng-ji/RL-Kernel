# H100 handoff: #543 / #544 (2026-10-29, v2)

**For:** the agent on the Hopper machine. Rules of `HANDOFF.md` (v2) and `HANDOFF-557-EP4-CHAIN.md` sections 2, 5, 6 apply unchanged (no push, no force, no PR edits, desensitize, `evidence_check.sh`, one UTC clock).
**Run this only after #557 (`HANDOFF-557-EP4-CHAIN.md`) is reported.**

## 1. Why

The Hopper sections of #543 and #544 report results measured with two local, unmerged changes: the Hopper allowance (`2a8f18e4`) and Megatron recompute loaded without DDP/optimizer (`pr_assets/hopper-qwen3-next/pr503/hopper_recompute_no_ddp.patch`). Wenbo has now written both as **local, unpushed commits** on top of the #543 head. This round measures (A) what the public heads do on 4 x H100 and (B) whether the two candidate commits make the full chain run, with an exact, checkable base.

## 2. Pinned inputs

| name | what | id |
|---|---|---|
| #543 public head | fork `feat/ws1-qwen3-next-full-model-chain` | `d78b22b3` |
| #544 public head | fork `feat/ws1-qwen3-next-batch-invariance-ci` | `db1eb37f` |
| candidate commits | `0001-...Hopper...patch`, `0002-...without-DDP...patch` in `543-hopper-patches/` (made with `git format-patch d78b22b3..`) | resulting HEAD `36ca09dc`, tree `3cd320745c0ea04b18b453e89ed75d914367c048` |

- Check the patch files against `543-hopper-patches/SHA256SUMS` before use.
- Head check is mandatory in every worktree. For B: `git rev-parse --short=8 HEAD` = `d78b22b3` **before** `git am`, then after `git am` `git rev-parse HEAD^{tree}` must equal the tree above. Anything else: stop and report.
- Fresh worktree, venv and build per run; `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 TORCH_CUDA_ARCH_LIST=9.0`; `_C.__file__` inside the worktree. Environment record as in the #557 handoff section 3.

## 3. Run A: public heads, no patch

| step | command | expected |
|---|---|---|
| unit + gates (#543) | `python -m pytest -q $(grep -oE "tests/models/qwen3_next/test_[a-z0-9_]+\.py" .github/workflows/ci.yml \| sort -u)`, then `TRITON_F32_DEFAULT=ieee python -m pytest -q tests/models/qwen3_next/check_qwen3_next_*.py tests/models/qwen3_next/check_gdn_recurrent_golden.py` | pass |
| unit (#544) | first two commands of the "Test" section of the #544 body, verbatim | pass |
| chain (#543) | `qwen3_next_run.py --stage all` with the arguments of the #543 body | expected to fail at `preflight`: `topology_mismatch: requires exactly four B200 GPUs`. Keep the failure text; do not patch |

## 4. Run B: #543 head + the two candidate commits (new worktree, `git am` the two files in order)

| step | command | what to record |
|---|---|---|
| unit | `python -m pytest -q tests/models/qwen3_next/test_qwen3_next_native_recompute.py tests/models/qwen3_next/test_qwen3_next_chain_report.py` | expect 25 passed (CPU result by Wenbo) |
| chain | `qwen3_next_run.py --stage all --until l2` with the arguments of the #543 body (TP4 / EP1) | stage results; `full_model_l2` outcome; per-GPU peak memory (`nvidia-smi --query-gpu=memory.used --format=csv` sampled every 10 s into a file) |
| training stages | only if `l2` passed: `train1`, `train2` as in the #543 body | training path is unchanged by the commits; this checks that |
| bi (#544 content) | only if `l2` passed: apply #544's commits on top **only if they apply cleanly**; otherwise stop and report. `qwen3_next_run.py --stage bi --previous <l2 run> --bi-suite quick` | cell counts |

- The commits change `create_actor(training=False)` only; the expected L2 outcome is the same as the published one (`full_model_l2: passed`). State the result as measured; do not compare to B200 numbers as if equal.
- If OOM or any failure: record stage, rank, `nvidia-smi` memory, error text (desensitized); stop. Do not change batch size, sequence length, memory fractions or code.

## 5. Not decided here (Wenbo)

Whether the two commits are pushed into #543 (this moves the heads of #543, #544, #549, #557, #551 and the evidence tied to their old SHAs). This round only produces measurements on a labelled, unpushed base. Report the base as "#543 head `d78b22b3` + two local candidate commits (tree `3cd32074…`)"; never call it a PR head.

## 6. Report

`report-543-544.md`: env table, head/tree checks, per step command/UTC/exit code/counts, memory log summary, exact preflight error text from A, list of any deviation (none expected). Deliver under `pending/h100-qwen3-next-543-544/` with `SHA256SUMS`. Do not push.
