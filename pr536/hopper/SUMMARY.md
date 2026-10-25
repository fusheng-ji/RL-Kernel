# PR #536 H100 summary

head: `54307e1b175943e91ad5f20379003ce73c4dd071`  
merge base: `16b81175874ecf11d5ff05c00de3b69e9457591b`  
4× H100-94GB, `RL_KERNEL_AUTOTUNE={0,1}`, private autotune cache (UUID/PCI redacted in cache dumps).

## pytest highlights (failures retained in logs)

| run | det_gemm | platform | tests/ops |
| --- | --- | --- | --- |
| head au=0 | 18 failed, 55 passed, 9 skipped | 514 passed, 29 skipped | 722 failed, 1324 passed, 401 skipped |
| head au=1 | 18 failed, 55 passed, 9 skipped | 514 passed, 29 skipped | 723 failed, 1323 passed, 401 skipped |
| base au=0 | 2 failed, 20 passed, 49 skipped | (see logs) | 244 failed, 1289 passed, 412 skipped |
| base au=1 | 2 failed, 20 passed, 49 skipped | (see logs) | 244 failed, 1289 passed, 412 skipped |

Many `tests/ops` failures follow CUDA illegal memory access after det_gemm stress (see unedited logs).

Benchmark: `issue50-performance.json` present.

Diff vs merge base on same box; do not cite absolute B200 counts.
