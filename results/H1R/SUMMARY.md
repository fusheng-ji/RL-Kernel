# H1 rerun on H100: `refs/h100/h3-strict-gemm-rows` (`775e98e7`) + the refiner test fix (2026-11-22)

**Verdict: PASS.** With the test-only fix applied, nothing fails:
- binding tests: 33 passed for the two-file command, 96 passed for the README's four files;
- the `tests/models/minimax_h3` suite at head: 1475 passed, 81 skipped.

`test_h3_token_refiner.py::test_backward_against_fp32_golden`, the only failure of the last round (`results/H1/` on `c57e23a1`), now passes.

## Code

- `<W1R>` is a detached worktree at `775e98e7`. `fix.patch` (this folder, byte for byte as given) was applied with `git am` (`git-am.txt`).
- `git rev-parse HEAD^{tree}` is `c6bf8c3e87782639741205d85fc99c5b9c500b28`, as required (`commit.txt`).
- `HEAD` is `f6d4ee71`, not the patch header's `02a0ef34`: `git am` sets a new committer and commit date. The tree is the same.
- `_C` was built in `<W1R>` with `<E1>` (`build.log`, rc 0). `rl_engine._C.__file__` was inside `<W1R>` after the build and before every step (`c-file*.txt`).
- `git status --porcelain --untracked-files=no` was empty before and after `git am`, the build and every step (`status-*.txt`).

## Runs

All runs were on node-a, one after another, with only our processes on the GPUs (`nvidia-smi-before-*.txt`, `other-processes-*.txt`). They used `<E1>`, `PYTHONPATH=<W1R>:<PROVIDER>/src` and `RL_KERNEL_H3_WEIGHTS=<WEIGHTS>`, as in H1.

| Step | GPUs | Output | Result |
|---|---|---|---|
| 2a, the requested command: `test_h3_one_block.py test_h3_token_refiner.py` | 0 | `binding-tests.txt` | 33 passed |
| 2a, the README H1 command: the same plus `test_h3_operator_trace.py test_h3_full_model.py` | 0 | `binding-tests-readme.txt` | 96 passed |
| 2b, `tests/models/minimax_h3` at head (`--tb=line`, as in H1) | 0,1 | `suite-head.txt` | 1475 passed, 81 skipped |

### Suite against the last round

| Suite | Passed | Failed | Skipped |
|---|---|---|---|
| `suite-base` (`9388feed`, last round) | 1470 | 0 | 81 |
| `suite-head` (`775e98e7`, last round) | 1474 | 1 | 81 |
| `suite-head` (`<W1R>`, this run) | 1475 | 0 | 81 |

- **The extra tests.** Collection at both commits (`collect-w1r.txt`, `collect-b1.txt`, on CPU) has 1556 against 1551 tests. The five tests that exist only at head (`tests-only-at-head.txt`) are PR 1's additions to `test_h3_one_block.py`:
  - `test_bindings_report_the_operators_that_ran`
  - `test_gemm_row_nodes_pass_logical_keys_and_a_parameter_id_per_block`
  - `test_gemm_rows_mode_is_checked`
  - `test_gemm_rows_run_their_operators_on_sm90`
  - `test_interim_gemm_ops_open_no_session`
- **Tests missing at head:** none.
- **Skips.** The skip lines are identical to the last round's `suite-base`.

## Deviations

1. **The binding-test count.** The requested command (two files) collects 33 tests at `775e98e7`, not 96. The 96 is the README H1 command's count over four files. As agreed, both were run: the two-file command into `binding-tests.txt` and the four-file command into `binding-tests-readme.txt`.
2. **The commit hash.** `HEAD` differs from the patch header (`f6d4ee71` against `02a0ef34`) because `git am` records its own committer and date. The tree hash matches.
3. **Test collection.** The CPU collection of both suites (`collect-*.txt`) is an extra check, not one of the requested commands.
4. **`<WEIGHTS>`** is the same directory as in the last round: `<SCRATCH>/h3ckpt`, which holds `hf_full/` and the pinned files. See deviation 1 of `results/SUMMARY.md`.
