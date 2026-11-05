# 3. Step time, strict vs provider (#575)

Tree: `<SRC>/pr575` (39cb7d3d). One H100, nothing else running on that GPU (timings from a shared GPU are not usable).

```bash
scripts/run_h100.sh speed <SRC>/pr575 <OUT>/03/speed   # h3_full_speed.py --sizes 12,925,4253,8349 --repeats 3
```

**Changes since the published table** (head `4f7c5d11`):
- `CandidateOps` is now built once, outside the timed region. Before, every strict repeat paid for registry lookups,
  CUDA initialisation and binding setup.
- Strict mode now binds the video projections (`proj_in` / `proj_out`) to the rows of #646.

This tree does **not** contain the follow-up's fused QKV call. That lands separately.

Published H100 numbers (median of 3; total = forward + backward with gradient checkpointing):

| tokens | provider total | strict total | ratio | strict peak GiB |
|---|---|---|---|---|
| 12 | 0.312 s | 1.3 s | 4.03x | 62.9 |
| 925 | 0.438 s | 5.0 s | 11.33x | 63.0 |
| 4253 | 1.709 s | 37.4 s | 21.88x | 63.5 |
| 8349 | 3.625 s | 126.4 s | 34.88x | 65.0 |

**Expected:**
- provider times about the same as before;
- strict times lower at small sizes, where the per-repeat setup mattered most;
- the large sizes roughly unchanged, apart from the video rows.

Nothing in the code gates this. Report the table as measured, plus the `attention_backend` the runner records
(`h3_full_attention` expected when `_H3_C` is built).

**Send back:** `full-speed.json`, the run log, and the table above with this run's numbers.
