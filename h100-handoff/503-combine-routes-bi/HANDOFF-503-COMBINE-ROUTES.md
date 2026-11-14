# H100 handoff: #503 `combine_routes` route-weight gradient (2026-11-14)

**For:** the agent on the Hopper machine. No push, no force, no PR edits, desensitize, one UTC clock.

## 1. Why

#503's H100 report (at head `d5253a5c`) showed `tests/models/qwen3_next/test_*.py` and `check_*.py`
all passing (322 passed) and the TP4 MoE gate passing on 4 ranks, but a direct probe found that
neither covers the route-weight gradient of `combine_routes`: the stock `torch.sum` reduction in
its backward made that gradient non-invariant from batch T = 16 up (T = 8 and 9 equal; output and
expert gradients equal at every T). Wenbo has now written the fix as a **local, unpushed commit**
on top of the #503 head: `combine_routes` becomes a `torch.autograd.Function` whose route-weight
gradient sums over H with the existing `fixed_order_row_sum` helper, plus a regression test. This
round re-runs the #503 suites and the probe at that commit on H100, with an exact, checkable base,
so the PR description can cite one SHA instead of a hand-applied patch.

## 2. Pinned inputs

| name | what | id |
|---|---|---|
| #503 public head | fork `feat/ws1-qwen3-next-moe-route-combine` | `d5253a5c` |
| candidate commit | `0001-fix-qwen3-next-fixed-order-route-weight-gradient-in.patch` in this directory (made with `git format-patch d5253a5c..`) | resulting HEAD `e514d42f` (the HEAD id may differ after `git am`; the tree must match), tree `ce3f499d` |
| touches | `rl_engine/models/qwen3_next/qwen3_next_forward.py`, `tests/models/qwen3_next/check_qwen3_next_forward.py`, `docs/operators/qwen3-next-moe-route-combine.md` | 3 files, +66/-18 |

- Check the patch file against `SHA256SUMS` before use.
- Head check is mandatory in every worktree. For B: `git rev-parse --short=8 HEAD` = `d5253a5c` **before** `git am`,
  then after `git am` `git rev-parse HEAD^{tree}` must equal `ce3f499d23552b6e081f69cc070a2813a8faede2`. Anything
  else: stop and report.
- Fresh worktree, venv and build per run; `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1
  TORCH_CUDA_ARCH_LIST=9.0`, `ninja` on `PATH`. `_C.__file__` must resolve inside that worktree
  before any run (an editable install elsewhere on `PATH`/`PYTHONPATH` will silently shadow it).
- Env: torch 2.13.0+cu130, triton 3.7.1, vllm 0.30.0, transformers 5.17.0, python 3.12,
  `VLLM_BATCH_INVARIANT=0`. 4 visible GPUs per run.

## 3. Run A: public head, no patch

| step | command | expected |
|---|---|---|
| unit + gates | `python -m pytest -q tests/models/qwen3_next/test_*.py tests/models/qwen3_next/check_*.py` | 322 passed, 0 failed, 0 skipped (matches the earlier H100 report) |
| TP4 MoE gate | the command recorded in `header.txt` / `tp_moe.log` under `hopper-qwen3-next/pr503/d5253a5c-nopatch/` on the fork's `pr_assets` branch | 4/4 ranks passed |
| probe | `python hopper-qwen3-next/pr503/d5253a5c-nopatch/probe_combine.py` from the same `pr_assets` path (copy it into the worktree first) | route-weight gradient equal at T = 8, 9; **not** equal at T = 16, 32, 64, 128, 256, 1024; output and expert gradients equal at every T |

## 4. Run B: #503 head + the candidate commit (new worktree, `git am` the one patch)

| step | command | what to record |
|---|---|---|
| unit + gates | same two test paths as Run A | expect 323 passed (one new test), 0 failed, 0 skipped |
| new regression test alone | `python -m pytest -q tests/models/qwen3_next/check_qwen3_next_forward.py -k combine_gradients` | expect pass |
| TP4 MoE gate | same command as Run A | 4/4 ranks passed |
| probe | same `probe_combine.py`, run from this worktree | bitwise equal for every T up to 1024 |

If Run A's suite does not reproduce 322/0/0, or Run B's probe is not equal at every T: stop and
report the exact failure; do not edit the patch or the tests to make it pass.

## 5. Not decided here (Wenbo)

Whether the commit is pushed into #503 (this moves the PR head and the SHA the existing
`pr_assets/hopper-qwen3-next/pr503/d5253a5c-*` evidence directories are tied to, and the
`d5253a5c-combine-routes-patch/` directory — the earlier hand-applied-patch run — becomes
redundant with Run B here but is not deleted by this round). Report the base as "#503 head
`d5253a5c` + one local candidate commit (tree `ce3f499d…`)"; never call it a PR head.

## 6. Report

`report-503-combine-routes.md`: env table (GPU model, driver, torch), head/tree checks for both
worktrees, per-step command/UTC/exit code/counts, the full probe output for A and B, and
`git status --short` of both worktrees. For any failure: `pytest -x --tb=long -rf` for that test.
Desensitize (no node names, job IDs, absolute paths, personal emails) before committing anything;
deliver results under `hopper-qwen3-next/pr503/<B's short SHA>/` on `pr_assets`, alongside the
existing `d5253a5c-*` directories (do not overwrite them). Do not push this branch's own tree.
