# H3 H100 remaining work (handoff `remaining-20261114`): what came back

Per item: `01/`, `02/`, `03/` (each with its own `SUMMARY.md`). Every run folder has `commit.txt` (tested SHA, clean
tree), `environment.txt`, `pip-freeze.txt` and the raw outputs. Hosts are `node-a` / `node-b`; user, cluster, job and
personal paths are replaced (`<SRC>`, `<OUT>`, `<ENV>`, `<user>`). **Nothing was pushed, no PR, no comment.**

| Item | Result | Local branch / SHA |
| --- | --- | --- |
| 01 follow-up PR | merge-up clean (`e3c4cbc9`), no overlap with the #569 / #575 fixes; tests 1114 / 3 skipped / 0 failed vs baseline 1102 / 3 / 0; H-C2 holds (strict 0/16, provider 16/16, seeds 11 22 33). **Finding:** strict `fulltrans` fails on the unchanged branch (`parameter gradients require canonical_backward_session`); measured with an extra fix, `a6c93313`. PR draft updated. | `fix/h3-merged-rows-review-v2` `e3c4cbc9`; `fix/h3-strict-video-projection-session` `a6c93313` |
| 02 time_embedder / norm_out | all 8 boundary modules `row` in strict; candidate bitwise repeatable; tests 1118 / 3 / 0; dual strict 0/16 on 3 seeds; numbers moved as expected | `feat/h3-strict-time-embedder-norm-out` `89cbd77f` (includes `a6c93313`) |
| 03 WS2 full model | `c1455a75` (as handed over): tests pass, **every grid runs out of memory** in the canonical run on a 94 GB H100. With four small commits of their own all 11 grids are bitwise equal to canonical (TP2/4/8, TP2xCP2 ring + Ulysses, TP4xCP2 ring, + SP, sizes 40 / 925 / 928) and the inject control is caught at block 17 (row 5; with SP the owner is rank 3 alone). | `feat/h3-ws2-full-model` `c1455a75`; `fix/h3-ws2-full-model-memory` `2fe3acb2` |
| 04 unclaimed rows | **not started**: each row needs a claim comment on #420, which only Wenbo posts, and I do not comment on GitHub. | |

## Decisions for Wenbo

1. `a6c93313` (session for the strict video projection rows): needed for any strict full-model run on `test-h3`. Which PR?
2. The WS2 memory fix (`ad6b1a6d`, `8db2cd98`; typing `79fd9218`, `2b838164`; docs `2fe3acb2`): needed for any grid to run;
   `8db2cd98` changes `rl_engine/validation/models/h3_tp.py` (leaf is a view of a contiguous shard, not a clone).
3. 15 pytest cases of item 03 need 5-8 GPUs on one node and are skipped on a 4-GPU node (covered by the two-node grids).
4. Item 04: post the claim comments if the rows should be started.

## Deviations from the expectations

- 01: strict `fulltrans` baseline failure (above). mypy with torch is not clean at the baseline either (36 errors; CI-style
  mypy without torch is clean everywhere).
- 02: the tested tree includes `a6c93313` by merge.
- 03: grids did not run as handed over; `tp4-cp2-ring-sp` at size 925 is refused by design (SP needs rows divisible by TP),
  so the uneven-CP-shard case was run without SP; TP group spanning nodes with CP > 1 needs 16 GPUs and was not run; the first
  attempt of the memory fix (`ad6b1a6d` alone) was not sufficient and its rank logs were overwritten by the rerun.
