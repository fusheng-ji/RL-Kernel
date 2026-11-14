# H100 handoff: the remaining H3 work (RFC #420, MiniMax-H3), 2026-11-14

Hardware: 2 nodes x 4 H100 (sm90). There is no 8-GPU node, so TP8 spans both nodes.
ROCm work is out of scope. Paths are placeholders (`<REPO>`, `<SRC>`, `<CKPT>`, `<WEIGHTS>`, `<PROVIDER>`, `<OVERLAY>`,
`<VLLM_OMNI>`, `<VERL_OMNI>`, `<OUT>`).

## Where things stand

- Base: `upstream/test-h3` @ `07533aee`. Every H3 PR from fusheng-ji is merged (#483–#494, #576, #586–#588, #567–#575; #569
  and #575 on 2026-11-09).
- Two local branches are not in `test-h3`, not on the fork and not in a PR. Both are in `remaining.bundle`:
  - `fix/h3-merged-rows-review` @ `0264a4f8`: six fixes for the review round of #568/#570/#571/#574, which were merged
    while that round was still open. The PR text draft is `followup-pr-draft.md` (in this folder, a copy of the 2026-11-05 draft).
  - `local/h3-ws2-fullmodel` @ `3f461dd9`: ring CP, distributed packed IO, CP weight gradients, and the full 50-block model
    on TP x CP x SP. Last touched on 2026-10-29. It has no H100 evidence.
- `git merge-tree` against `07533aee` reports no textual conflict for either branch. The overlaps that must still be
  checked by hand are listed in items 01 and 03.
- The 2026-11-05 handoff (fork branch `handoff/h3-h100-review-followup`, `h100-handoff/review-followup-20261105/`) never came back. Its item 1a and item 2 are folded into
  item 01 here. Its #569 and #575 parts are no longer needed (both PRs are merged).
- Strict full-model runs only work on sm90: the video projection rows (#646) require `cc == (9, 0)`. On B200 they stop at
  `proj_in`.

## Code: one bundle, two branches

The bundle needs two public commits, both in `test-h3`: `cfec0d98` (base of the follow-up) and `3baf93c8` (base of the
WS2 branch).

```bash
cd <REPO>                                  # clone of RL-Align/RL-Kernel, remote "upstream"
git fetch upstream test-h3                 # must reach 07533aee
git bundle verify <THIS_DIR>/remaining.bundle
git fetch <THIS_DIR>/remaining.bundle \
    'refs/heads/fix/h3-merged-rows-review:refs/heads/fix/h3-merged-rows-review' \
    'refs/heads/local/h3-ws2-fullmodel:refs/heads/local/h3-ws2-fullmodel'
git rev-parse fix/h3-merged-rows-review local/h3-ws2-fullmodel   # 0264a4f8... and 3f461dd9...
git worktree add --detach <SRC>/base upstream/test-h3            # 07533aee, the baseline for the test suite
```

Items 01–03 each create their own branch from these. Merge `test-h3` into the branch (merge-up). **Do not rebase**:
the stack convention is merge-up, and rebasing would rewrite signed commits.

## Environment (same as the earlier H100 sections)

- torch 2.13.0+cu130, CUDA 13.0, the pinned `MiniMax-H3@42ed227` checkpoint (`<CKPT>/transformer`, 14 shards).
  `RL_KERNEL_H3_WEIGHTS` points at `<WEIGHTS>` (`h3_conditioning.safetensors`, `h3_block0.safetensors`,
  `hf/transformer` -> the 14 shards).
- Provider: diffusers at `f53d552036a0d1bd5570782a39cd40cfabf112bc`, clean checkout (`<PROVIDER>`). The runners check it.
- Optional, pinned as before: vLLM-Omni `12e92800`, verl-omni (`VERL_OMNI_PATH`, for the scheduler comparison test).
- Build the extensions **on the H100**, in each worktree, with `RL_KERNEL_H3_STRICT=1`. Then check that
  `rl_engine._C.__file__` and `rl_engine._H3_C.__file__` point into that worktree: the editable `_C` is shared, and
  resolves to whichever worktree ran `pip install -e` last.
- `scripts/run_h100.sh KIND SRC OUT [ARGS]` does the environment, the build and the checks. Its stages can be resumed.
  Kinds: `build`, `tests`, `mypy`, `fulltrans`, `dual`, `ws2full`. Set `PY`, `CKPT`, `WEIGHTS` and `PROVIDER` first.

## Items

| # | File | GPUs | What it settles |
|---|---|---|---|
| 01 | `01-followup-pr.md` | 4, then 1 + 1 | follow-up PR: merge-up, overlap check, test suite + mypy, 33B strict vs provider, rollout vs train 0/16 |
| 02 | `02-bind-time-embedder-norm-out.md` | 1 | the last two boundary modules still on the provider in strict mode (`time_embedder`, `norm_out`) |
| 03 | `03-ws2-full-model.md` | 4 and 8 (2 nodes) | ring CP, CP weight gradients, the full model on TP x CP x SP; RFC row `final_distributed_h3_run` |
| 04 | `04-unclaimed-rows.md` | as needed | the RFC rows nobody has claimed that need no ROCm hardware and no one else's rows |

Order: 01 → 02 (on top of 01) → 03 (independent of 01/02, can run on the other node).

## Rules (unchanged)

- No push, no PR and no comment on GitHub until Wenbo says so. Evidence goes to `pr_assets` only after "push evidence".
- Never change a test bound or a runner to make a result pass. A failure is reported as a failure, with its log.
- Anything that may reach a PR is redacted: no cluster, job, node, partition, user or personal paths. Hosts become
  `node-a` / `node-b`. The WS2 report records `socket.gethostname()` for every rank, so replace those values before upload.
- Commits use `git commit -s` and carry no AI attribution lines (no `Co-Authored-By`, no "Generated with").
- Before pushing anything that touches `rl_engine/`, run `mypy rl_engine`. CI lint runs mypy, and a mypy error skips the
  unit tests.

## What to send back

One folder per item, with:
- `commit.txt` (the tested SHA, clean tree);
- `environment.txt` (`nvidia-smi`, torch/CUDA/triton, the `_C` / `_H3_C` paths, the provider commit) and `pip-freeze.txt`;
- the raw outputs (`*.json`, junit `*.xml`, logs);
- a short `SUMMARY.md` with the numbers asked for in the item file and any deviation from the expectation.
