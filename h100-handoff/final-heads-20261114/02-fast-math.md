# 02. The fast-math build (`KERNEL_ALIGN_USE_FAST_MATH=1`)

The H3 stack's Tests tables (#569, #571) have a fast-math row next to the default build. None of #699 / #700 / #701 has
one yet. #701 changes `csrc/cuda/attention/deterministic_attention.cu`, so it matters most there.

The fast-math build leaves the H3 kernels out, so most H3 kernel tests skip. On #569 / #571 the fast-math row was
961 / 611 / 2 (passed / skipped / failed). That is expected, and it is why a `test-h3` baseline in the same build is
needed: a failure counts only if the baseline does not have it.

Each worktree gets its own in-place build. Use the `-fm` worktrees from the README, never the default-build ones.

```bash
G=scripts/run_h100.sh
FAST_MATH=1 $G tests <SRC>/base-fm  <OUT>/02/base-fm      # test-h3 07533aee, the baseline
FAST_MATH=1 $G tests <SRC>/pr699-fm <OUT>/02/pr699-fm
FAST_MATH=1 $G tests <SRC>/pr700-fm <OUT>/02/pr700-fm
FAST_MATH=1 $G tests <SRC>/pr701-fm <OUT>/02/pr701-fm
```

One node, 4 GPUs visible. `environment.txt` must show `KERNEL_ALIGN_USE_FAST_MATH 1`, and `_H3_C` either absent or built
from that worktree.

**Expected:**
- **New failures:** none in a PR tree that `base-fm` does not also have.
- **Known failure:** on #569 / #571 the fast-math build failed
  `test_h3_final_adaln_out.py::TestCuda::test_unsupported_device_is_refused_at_construction`. With the H3 kernels left
  out, the constructor raises "lacks the h3_det_linear_* / h3_rmsnorm_* ops" before the SM80 check the test expects.
  If it fails here too, it belongs to `test-h3` and is not a PR finding.

## Send back

`02/<tree>/`, and a `SUMMARY.md` table:

| Tree | Commit | passed / skipped / failed | Failures not in `base-fm` |
|---|---|---|---|
| base-fm | `07533aee` | | |
| pr699-fm | `0566cdbb` | | |
| pr700-fm | `3eda7d4f` | | |
| pr701-fm | `e42c763a` | | |

For each failure that is not in `base-fm`, give its traceback.
