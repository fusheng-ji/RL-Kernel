# 2. 33B strict vs provider, and rollout vs train (RFC H-C2)

Tree: `<SRC>/followup` (0264a4f8). This is `test-h3` (#571 merged, with the video projections bound in strict mode)
plus the follow-up fixes, the fused QKV backward among them. One H100 per runner.

```bash
scripts/run_h100.sh fulltrans <SRC>/followup <OUT>/02/fulltrans   # h3_full_transformer.py, seeds 11 22 33
scripts/run_h100.sh dual      <SRC>/followup <OUT>/02/dual        # h3_dual_trajectory.py,  seeds 11 22 33
```

## 2a. `h3_full_transformer.py` (strict vs provider, forward and backward, tiny request)

For each seed, record:
- candidate bitwise repeatable, provider repeatable;
- the video and audio output relative differences;
- the input-gradient relative differences (video hidden states, audio, encoder);
- the parameter-gradient median and maximum;
- peak memory.

The last H100 numbers in #571 were taken before the video projection binding and before the fused QKV backward (head
`5ae9f5eb`): video 2.07e-03, audio 1.77e-02, input gradients 0.23 / 0.23 / 0.33.

**Expected:**
- the candidate is still bitwise repeatable;
- the output differences move, because `proj_in` / `proj_out` now run the strict rows instead of the provider's cuBLAS;
- the input-gradient columns move, because of the fused QKV backward.

There is no committed bound on these columns: report the numbers. If the candidate is **not** repeatable, that is a
failure. Report it with the log.

## 2b. `h3_dual_trajectory.py` (rollout single vs train batched, 4 Euler steps x 4 comparisons)

The H100 table in #571 (before the binding):

| Seed | Provider | Strict | Strict + video projections bound (diagnostic, now removed) |
|---|---|---|---|
| 11 | 16/16 | 8/16 | 0/16 |
| 22 | 16/16 | 8/16 | 0/16 |
| 33 | 16/16 | 8/16 | 0/16 |

The diagnostic column bound the video projections to the generic `h3_projection` GEMM. Strict mode now binds the
dedicated rows of #646 itself.

**Expected (RFC H-C2):** strict is **0/16** differing comparisons on all three seeds, and the provider stays 16/16. A strict
result that is not 0/16 is a finding, not something to tune. Record:
- the first differing step and tensor;
- the max abs difference per step;
- whether video, audio or both differ.

`h3_dual_trajectory.py --help` lists the arms. There is no `--diag-video-projections` any more.

**Send back:** the per-seed JSON from both runners and a `SUMMARY.md` table in the same layout as above, with this run's
numbers.
