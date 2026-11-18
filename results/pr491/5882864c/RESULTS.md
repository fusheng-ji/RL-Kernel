# #491 at `5882864c`, before = `upstream/main` `49941801`

- GPU: 4 x NVIDIA H100 per node (95830 MiB each), driver 580.173.02, CUDA 13.0 (toolkit 13.x), torch 2.13.0+cu130, triton 3.7.1, python 3.12.15, vLLM 0.30.0. Exact per-run values: `header.txt`. Topology (`nvidia-smi.txt`): the 4 GPUs of a node are pairwise connected by NVLink (NV6); no PCIe-only GPU pair.
- Build: `TORCH_CUDA_ARCH_LIST=9.0`, `MAX_JOBS=32`, `ninja` on `PATH`. Before each GPU run `rl_engine._C.__file__` was printed and is inside the worktree of the run (`c-file.txt` / `steps.txt`).
- `git status --porcelain --untracked-files=no` was empty before (`header.txt`) and after (`status-after.txt`, empty) every run. No file in a worktree was edited; no patch was applied to any run in this directory.
- The shared environment was not modified. `ninja` (and, for a diagnostic of task A only, a different `apache-tvm-ffi`) come from separate directories.
- Node: the nodes were exclusive to this work. On node 1 two runs (#495 on GPUs 0-1, #538 on GPUs 2-3) ran at the same time in separate one-pair steps; all other runs were alone on their node, and no foreign process was visible (`header.txt`).

## Commands (`steps.txt`)
- `TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace` on the head and (earlier, by the #536 run) on `upstream/main` `49941801`, same flags.
- `python -m pytest -p no:cacheprovider -v -rs tests/distributed/collectives/ tests/benchmarks/` (`pytest.log`)
- Back to back on the same GPUs, with **this head's** benchmark script and `PYTHONPATH` pointing at the main worktree for "before": `python -m torch.distributed.run --standalone --nproc-per-node=N benchmarks/distributed/benchmark_deterministic_collectives.py --output {before,after}_wN.json` for N = 2 (GPUs 0,1) and N = 4. (`torchrun` of the shared interpreter would pick another python, hence `python -m torch.distributed.run`.)
- **No 8-GPU run**: the nodes of this allocation have 4 GPUs. A two-node 8-rank run would measure the inter-node network, not the single-node collectives of this PR, so it was not done.

## Results
### pytest.log

Result line: `94 passed, 6 skipped in 21.14s`

Skip reasons (`-rs`):

- 3 x requires eight visible CUDA GPUs
- 3 x requires eight visible Ascend NPUs


Benchmark runs: all four exit codes 0 (`steps.txt`). Per-case table (median-based `deterministic_us`, from the JSON; main = before, head = after; `before-after-table.md`):

| GPUs | operation | input bytes / rank | main (before) us | head (after) us | after / before | NCCL us (after run) |
|---|---|---|---|---|---|---|
| 2 | all_gather | 1024 | 29.1 | 26.1 | 0.90 | 57.7 |
| 2 | all_gather_many | 1024 | 67.9 | 68.5 | 1.01 | 86.0 |
| 2 | reduce_scatter | 2048 | 30.4 | 31.2 | 1.03 | 61.1 |
| 2 | all_reduce | 1024 | 24.1 | 25.2 | 1.04 | 68.8 |
| 2 | all_gather | 8192 | 56.0 | 27.0 | 0.48 | 57.9 |
| 2 | all_gather_many | 8192 | 67.8 | 68.2 | 1.00 | 83.5 |
| 2 | reduce_scatter | 16384 | 37.4 | 38.3 | 1.03 | 64.8 |
| 2 | all_reduce | 8192 | 25.4 | 25.6 | 1.01 | 69.9 |
| 2 | all_gather | 16384 | 86.9 | 29.3 | 0.34 | 59.1 |
| 2 | all_gather_many | 16384 | 67.1 | 68.3 | 1.02 | 85.4 |
| 2 | reduce_scatter | 32768 | 48.0 | 48.1 | 1.00 | 62.7 |
| 2 | all_reduce | 16384 | 25.1 | 25.3 | 1.01 | 67.4 |
| 2 | all_gather | 32768 | 148.6 | 33.9 | 0.23 | 57.9 |
| 2 | all_gather_many | 32768 | 66.9 | 68.6 | 1.02 | 87.5 |
| 2 | reduce_scatter | 65536 | 68.7 | 68.8 | 1.00 | 65.0 |
| 2 | all_reduce | 32768 | 26.4 | 26.3 | 1.00 | 69.4 |
| 2 | all_gather | 65536 | 273.1 | 35.6 | 0.13 | 58.2 |
| 2 | all_gather_many | 65536 | 67.1 | 68.0 | 1.01 | 85.3 |
| 2 | reduce_scatter | 131072 | 110.2 | 110.1 | 1.00 | 64.8 |
| 2 | all_reduce | 65536 | 27.4 | 27.7 | 1.01 | 67.9 |
| 2 | all_gather | 98304 | 397.0 | 36.7 | 0.09 | 59.1 |
| 2 | all_gather_many | 98304 | 67.9 | 67.7 | 1.00 | 84.3 |
| 2 | reduce_scatter | 196608 | 152.1 | 151.8 | 1.00 | 68.4 |
| 2 | all_reduce | 98304 | 31.4 | 29.8 | 0.95 | 68.7 |
| 2 | all_gather | 131072 | 520.7 | 38.8 | 0.07 | 58.7 |
| 2 | all_gather_many | 131072 | 68.6 | 68.4 | 1.00 | 84.5 |
| 2 | reduce_scatter | 262144 | 192.9 | 192.6 | 1.00 | 68.2 |
| 2 | all_reduce | 131072 | 32.4 | 31.9 | 0.98 | 69.1 |
| 2 | all_gather | 196608 | 44.0 | 41.6 | 0.95 | 58.8 |
| 2 | all_gather_many | 196608 | 69.3 | 68.7 | 0.99 | 84.4 |
| 2 | reduce_scatter | 393216 | 49.5 | 48.9 | 0.99 | 67.7 |
| 2 | all_reduce | 196608 | 35.5 | 36.0 | 1.01 | 68.4 |
| 2 | all_gather | 262144 | 47.7 | 44.4 | 0.93 | 59.8 |
| 2 | all_gather_many | 262144 | 69.4 | 68.1 | 0.98 | 85.7 |
| 2 | reduce_scatter | 524288 | 49.7 | 48.3 | 0.97 | 65.0 |
| 2 | all_reduce | 262144 | 40.4 | 39.7 | 0.98 | 68.8 |
| 2 | all_gather | 524288 | 51.4 | 47.2 | 0.92 | 62.3 |
| 2 | all_gather_many | 524288 | 70.0 | 70.1 | 1.00 | 91.2 |
| 2 | reduce_scatter | 1048576 | 54.0 | 53.1 | 0.98 | 83.3 |
| 2 | all_reduce | 524288 | 48.1 | 47.9 | 1.00 | 73.2 |
| 2 | all_gather | 1048576 | 58.2 | 52.8 | 0.91 | 75.7 |
| 2 | all_gather_many | 1048576 | 72.5 | 72.5 | 1.00 | 90.1 |
| 2 | reduce_scatter | 2097152 | 56.1 | 55.4 | 0.99 | 73.9 |
| 2 | all_reduce | 1048576 | 51.3 | 52.2 | 1.02 | 73.8 |
| 2 | all_gather | 4194304 | 105.5 | 83.8 | 0.79 | 101.6 |
| 2 | all_gather_many | 4194304 | 96.6 | 97.1 | 1.01 | 118.0 |
| 2 | reduce_scatter | 8388608 | 84.2 | 84.6 | 1.00 | 101.7 |
| 2 | all_reduce | 4194304 | 82.0 | 80.9 | 0.99 | 98.5 |
| 4 | all_gather | 1024 | 46.2 | 32.4 | 0.70 | 67.1 |
| 4 | all_gather_many | 1024 | 70.4 | 81.3 | 1.15 | 87.0 |
| 4 | reduce_scatter | 4096 | 33.3 | 32.2 | 0.97 | 68.3 |
| 4 | all_reduce | 1024 | 24.6 | 32.4 | 1.32 | 69.3 |
| 4 | all_gather | 8192 | 102.5 | 39.8 | 0.39 | 70.0 |
| 4 | all_gather_many | 8192 | 70.7 | 83.9 | 1.19 | 100.0 |
| 4 | reduce_scatter | 32768 | 43.3 | 48.0 | 1.11 | 68.6 |
| 4 | all_reduce | 8192 | 25.1 | 29.9 | 1.19 | 65.2 |
| 4 | all_gather | 16384 | 178.1 | 38.9 | 0.22 | 55.3 |
| 4 | all_gather_many | 16384 | 71.8 | 76.6 | 1.07 | 86.9 |
| 4 | reduce_scatter | 65536 | 52.5 | 53.2 | 1.01 | 62.0 |
| 4 | all_reduce | 16384 | 24.6 | 31.1 | 1.27 | 65.5 |
| 4 | all_gather | 32768 | 329.4 | 35.7 | 0.11 | 55.9 |
| 4 | all_gather_many | 32768 | 71.2 | 81.2 | 1.14 | 88.3 |
| 4 | reduce_scatter | 131072 | 75.6 | 75.1 | 0.99 | 61.5 |
| 4 | all_reduce | 32768 | 25.9 | 30.0 | 1.16 | 64.8 |
| 4 | all_gather | 65536 | 631.8 | 38.8 | 0.06 | 58.3 |
| 4 | all_gather_many | 65536 | 73.8 | 78.5 | 1.06 | 89.5 |
| 4 | reduce_scatter | 262144 | 120.1 | 120.2 | 1.00 | 62.4 |
| 4 | all_reduce | 65536 | 27.9 | 30.3 | 1.08 | 64.8 |
| 4 | all_gather | 98304 | 40.8 | 42.9 | 1.05 | 57.5 |
| 4 | all_gather_many | 98304 | 77.7 | 78.7 | 1.01 | 86.2 |
| 4 | reduce_scatter | 393216 | 44.9 | 54.4 | 1.21 | 68.0 |
| 4 | all_reduce | 98304 | 31.2 | 39.6 | 1.27 | 65.2 |
| 4 | all_gather | 131072 | 42.0 | 47.6 | 1.13 | 57.2 |
| 4 | all_gather_many | 131072 | 79.8 | 81.2 | 1.02 | 86.3 |
| 4 | reduce_scatter | 524288 | 44.7 | 52.5 | 1.18 | 70.1 |
| 4 | all_reduce | 131072 | 32.2 | 40.8 | 1.27 | 65.0 |
| 4 | all_gather | 196608 | 47.3 | 49.8 | 1.05 | 57.5 |
| 4 | all_gather_many | 196608 | 82.4 | 86.9 | 1.06 | 87.9 |
| 4 | reduce_scatter | 786432 | 43.8 | 48.7 | 1.11 | 68.0 |
| 4 | all_reduce | 196608 | 35.9 | 43.7 | 1.22 | 65.7 |
| 4 | all_gather | 262144 | 53.9 | 50.4 | 0.93 | 58.4 |
| 4 | all_gather_many | 262144 | 85.3 | 86.1 | 1.01 | 85.5 |
| 4 | reduce_scatter | 1048576 | 46.5 | 54.3 | 1.17 | 66.4 |
| 4 | all_reduce | 262144 | 39.6 | 45.5 | 1.15 | 67.5 |
| 4 | all_gather | 524288 | 51.1 | 48.9 | 0.96 | 61.6 |
| 4 | all_gather_many | 524288 | 90.9 | 92.4 | 1.02 | 85.3 |
| 4 | reduce_scatter | 2097152 | 47.5 | 54.7 | 1.15 | 65.9 |
| 4 | all_reduce | 524288 | 41.9 | 51.1 | 1.22 | 69.4 |
| 4 | all_gather | 1048576 | 56.9 | 57.9 | 1.02 | 93.9 |
| 4 | all_gather_many | 1048576 | 104.6 | 104.3 | 1.00 | 107.9 |
| 4 | reduce_scatter | 4194304 | 51.7 | 55.3 | 1.07 | 96.9 |
| 4 | all_reduce | 1048576 | 48.2 | 57.4 | 1.19 | 68.3 |
| 4 | all_gather | 4194304 | 123.0 | 129.5 | 1.05 | 111.5 |
| 4 | all_gather_many | 4194304 | 194.3 | 194.9 | 1.00 | 126.3 |
| 4 | reduce_scatter | 16777216 | 90.7 | 90.9 | 1.00 | 106.1 |
| 4 | all_reduce | 4194304 | 78.0 | 80.1 | 1.03 | 83.0 |

Skips: 3 tests need eight visible CUDA GPUs and 3 need eight Ascend NPUs (not available on 4-GPU nodes).

## Topology (neutral words)
Single node, 4 GPUs, every GPU pair connected by NVLink (NV6 in `nvidia-smi topo -m`); no PCIe-only pair. CPU NUMA: two GPUs per NUMA node.

## 64 KiB single-block threshold
This run did not retune anything and did not run a threshold sweep beyond the script's default sweep, which straddles the 64 KiB / 256 KiB thresholds; the rows around those sizes are in the table. Whether the threshold "still holds" is a judgment on those rows that is left to the reviewer: the numbers are above and in the JSON.

## Draft "H100" paragraph
Single node, 4 x H100 (NVLink between every pair; driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130), head `5882864c`, build `TORCH_CUDA_ARCH_LIST=9.0`, detached clean worktree, no patch: `tests/distributed/collectives/` + `tests/benchmarks/` report 94 passed, 0 failed, 6 skipped (3 need eight visible CUDA GPUs, 3 need eight Ascend NPUs; nodes here have 4 GPUs). The benchmark of this head was run back to back with `upstream/main` `49941801` (before) on 2 and 4 GPUs; the full table is in the results. No 8-GPU node was available (nodes have 4 GPUs).
