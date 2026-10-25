# PR #576 H100 summary

head: ae8dff68413715fd75e0bd9c4a07c21b1667805e
pytest: 47 passed, 0 failed, 0 skipped

| seq | flash_v1 fwd+bwd (ms) | sdpa fwd+bwd (ms) | ratio flash/sdpa |
| --- | --- | --- | --- |
| 12 | 0.562 | 0.371 | 1.51x |
| 925 | 0.867 | 0.473 | 1.83x |
| 4096 | 7.164 | 3.131 | 2.29x |
| 8349 | 30.328 | 12.619 | 2.40x |

Across measured seq lengths, flash_v1 forward+backward is about **1.51x–2.40x** slower than SDPA on H100-94GB (median of three runs). Byte-level flash output repeat/batch/head-shard invariance holds within H100 only; mma.sync path may differ from B200 bytes.
