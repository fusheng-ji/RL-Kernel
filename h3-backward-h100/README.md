# Backward on Hopper (H100, sm90) for #484 to #489

For each op, gradients of three implementations on identical inputs and upstream gradients: the native diffusers replay, the same replay with `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` (a deterministic diffusers reference), and the RL-Kernel op. Every run is executed twice to test repeatability. Errors are max|err| / max|ref| against an FP64 autograd of the reference formula (available for S <= 4097).

* `<op>__summary.json`: per size and per gradient: repeatability of each implementation, whether the kernel equals the deterministic diffusers gradient bitwise, the fraction of differing elements, and the three golden errors.
* `bwd_det_ref_report.md`: findings. `det_diffusers_backward.py`: the runner.

Measured on NVIDIA H100 (sm90), torch 2.13.0+cu130, at the PR heads: #484 `ac66da29`, #485 `cb514529`, #486 `fb052a86`, #487 `f027e802`, #488 `7d5c8d1c`, #489 `e121939d`.

## 30-repeat native-diffusers repeatability (`native-repeat30/`)

The two-run repeat check above is extended to 30 replays per op and size, for native diffusers, deterministic diffusers and the rl-kernel op, with the CUDA kernels of one native backward listed. See `native-repeat30/REPORT.md`. Result: every gradient that flows through a diffusers `index_select` gather has 30 distinct bitwise results in native mode (run-to-run spread up to 0.2 of max|grad| at 32768 rows), and the native backward of those ops launches the atomic `indexFuncLargeIndex` (index_add) kernel. Deterministic diffusers and rl-kernel give 1 distinct result everywhere. Plain GEMM, layer-norm and element-wise gradients are repeatable in all three.
