# H100 handoff: the final heads of #699 / #700 / #701, fast math, and the missing WS2 layouts (2026-11-14)

Hardware: 2 nodes x 4 H100 (sm90), the same machines as the 2026-11-14 run. Everything here fits on them: nothing needs a
single 8-GPU node or more than 8 GPUs. ROCm is out of scope. Paths are placeholders (`<REPO>`, `<SRC>`, `<CKPT>`,
`<WEIGHTS>`, `<PROVIDER>`, `<VERL_OMNI>`, `<OUT>`).

## Why

The H100 evidence in the three draft PRs was taken at earlier commits. The fixes added afterwards have only run as CPU
tests on a B200 node:

| PR | Branch | Head to test | H100 evidence so far | Not run on a GPU since |
|---|---|---|---|---|
| #699 | `fix/h3-merged-rows-review-v2` | `0566cdbb` | suite at `e3c4cbc9`, strict full transformer at `a6c93313` | `0566cdbb`: an error inside the strict context propagates as it is |
| #700 | `feat/h3-strict-time-embedder-norm-out` | `3eda7d4f` | everything at `89cbd77f` | `ea4976c2` (SiLU check), `8557fef6` (strict mode requires the CUDA ops of `timestep_mlp_fp32` / `final_adaln_out`), two merges |
| #701 | `feat/h3-ws2-full-model` | `e42c763a` | grids at `8db2cd98`, suite at `2fe3acb2` | `3120ea58` (`shard_leaf` refactor), `e42c763a` (the shard check now needs 350 digests) |

The riskiest is #700's `8557fef6`: if the registry hands back anything but the CUDA class on the H100, strict mode refuses
to run.

## Code

The branches are public on the fork. No bundle is needed.

```bash
cd <REPO>                                  # clone of RL-Align/RL-Kernel, remote "upstream"
git fetch upstream test-h3                 # 07533aee
git fetch https://github.com/fusheng-ji/RL-Kernel.git \
    fix/h3-merged-rows-review-v2:refs/heads/fix/h3-merged-rows-review-v2 \
    feat/h3-strict-time-embedder-norm-out:refs/heads/feat/h3-strict-time-embedder-norm-out \
    feat/h3-ws2-full-model:refs/heads/feat/h3-ws2-full-model
git rev-parse --short fix/h3-merged-rows-review-v2 feat/h3-strict-time-embedder-norm-out feat/h3-ws2-full-model
#   0566cdbb  3eda7d4f  e42c763a   (stop if they differ: the PRs moved; ask before testing another head)
for t in 699:fix/h3-merged-rows-review-v2 700:feat/h3-strict-time-embedder-norm-out 701:feat/h3-ws2-full-model; do
  git worktree add --detach <SRC>/pr${t%%:*}      ${t#*:}     # default build
  git worktree add --detach <SRC>/pr${t%%:*}-fm   ${t#*:}     # fast-math build (item 02); builds are in place
done
git worktree add --detach <SRC>/base-fm upstream/test-h3      # fast-math baseline (item 02)
```

## Environment (same as the 2026-11-14 run)

- **Stack and checkpoint:** torch 2.13.0+cu130, CUDA 13.0, triton 3.7.1, the pinned `MiniMax-H3@42ed227` checkpoint
  (`<CKPT>/transformer`, 14 shards). `RL_KERNEL_H3_WEIGHTS` = `<WEIGHTS>`.
- **Provider:** diffusers at `f53d552036a0d1bd5570782a39cd40cfabf112bc`, a clean checkout. `VERL_OMNI` set as before,
  for the scheduler comparison in `test_h3_runtime_integration.py`.
- **Runner:** `scripts/run_h100.sh KIND SRC OUT [ARGS]`, the same script as last time with three changes:
  - `FAST_MATH=1` selects the fast-math build;
  - `mypy` runs as CI does (`--ignore-missing-imports rl_engine/`);
  - the environment check tolerates a build without `_H3_C`.

  Kinds are `build`, `tests`, `mypy`, `fulltrans`, `dual` and `ws2full`. Set `PY`, `CKPT`, `WEIGHTS` and `PROVIDER`
  first. The stages can be resumed.
- **Builds:** build in each worktree on the H100, then check `_C.__file__` in `environment.txt`. The editable `_C` is
  shared and resolves to the last `pip install -e`. The script builds in place and does not pip-install.

## Items

| # | File | GPUs | What it settles |
|---|---|---|---|
| 01 | `01-final-heads.md` | 1 node (4) + 2 nodes (8) | the three heads: suite, strict full transformer, dual trajectory, one WS2 grid and the SP negative control |
| 02 | `02-fast-math.md` | 1 node (4) | the `KERNEL_ALIGN_USE_FAST_MATH=1` row of each Tests table, against a `test-h3` baseline |
| 03 | `03-ws2-extra-layouts.md` | 1 node (3) + 2 nodes (8) | full-model TP4 x CP2 Ulysses, and ring CP3 |

Order: 01 first, because item 03 is only meaningful if #701's head passes 01. Item 02 can run on the other node at the
same time.

## Rules (unchanged)

- **Results:** never change a test bound, a runner or the code to make a result pass. A failure is reported as a
  failure, with its log. If a head fails, stop that PR's items and report.
- **GitHub:** no push, no PR edit, no comment.
- **Redaction:** anything that may leave the machine is redacted. No cluster, job, node, partition, user or personal
  paths. Hosts become `node-a` / `node-b`; the WS2 `report.json` records `socket.gethostname()` per rank.
- **Commits:** if a commit is needed, use `git commit -s` and add no AI attribution lines. None should be needed:
  this handoff only measures.

## What to send back

- **Where:** a fork branch `handoff/h3-h100-final-heads-20261114` with one folder `h100-results/final-heads-20261114/`.
- **Contents:** one sub-folder per item and run, holding `commit.txt`, `environment.txt`, `pip-freeze.txt`,
  `status.txt`, the raw outputs (`*.json`, junit `*.xml`, logs) and a `SUMMARY.md` with the tables each item asks for.
- **Push:** only that branch, after Wenbo's OK.
