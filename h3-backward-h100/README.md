# Backward on Hopper (H100, sm90) for #484 to #489

For each op, gradients of three implementations on identical inputs and upstream gradients: the native diffusers replay, the same replay with `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` (a deterministic diffusers reference), and the RL-Kernel op. Every run is executed twice to test repeatability. Errors are max|err| / max|ref| against an FP64 autograd of the reference formula (available for S <= 4097).

* `<op>__summary.json`: per size and per gradient: repeatability of each implementation, whether the kernel equals the deterministic diffusers gradient bitwise, the fraction of differing elements, and the three golden errors.
* `bwd_det_ref_report.md`: findings. `det_diffusers_backward.py`: the runner.

Measured on NVIDIA H100 (sm90), torch 2.13.0+cu130, at the PR heads: #484 `ac66da29`, #485 `cb514529`, #486 `fb052a86`, #487 `f027e802`, #488 `7d5c8d1c`, #489 `e121939d`.
