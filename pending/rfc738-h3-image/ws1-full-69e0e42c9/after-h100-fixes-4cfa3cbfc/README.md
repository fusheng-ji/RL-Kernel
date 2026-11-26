# B200 at `4cfa3cbfc` (after the two H100 fixes)

`d3bc9fe41` (contiguous rows into the PDD head projection) and `f4171b776` (`ffn_down` row run in slices of at most 32,768 rows without a weight gradient), merged into `feat/h3-image-ws1-full` at `4cfa3cbfc`. Clean tree, strict build, one B200.

- `pytest-suite.txt`: `tests/models/minimax_h3_image`, 535 passed (531 at `69e0e42c9`, plus three row-slice tests and one head test).
- `t2i-vs-69e0e42c9.summary.json`, `t2i_upscaled.records.json`: the strict `t2i_upscaled` trajectory in a new process against `../t2i/run-a` (`69e0e42c9`): `equal_to_other_process` true. On B200 the down row is interim and the head rows were already contiguous, so the bytes are unchanged.
