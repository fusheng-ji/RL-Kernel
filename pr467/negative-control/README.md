Negative control for the plain-op parameter-VJP change (PR #467).
Environment: 1x B200, torch 2.12.0+cu130 (conda env fastvideo), CUDA 13.0, extension built from `d73120c1` with TORCH_CUDA_ARCH_LIST=10.0.
"old implementation" = `rstd = torch.rsqrt(x.float().square().mean(dim=-1) + eps)` in `RMSNormCudaOp.parameter_vjp_contributions_fp32`, applied to a scratch checkout.

| test | current | old implementation |
|---|---|---|
| `test_ws1_gtest_gpu.py::test_c4_cuda_rms_norm_is_bitwise_invariant` | pass (`current.log`) | pass (`old-implementation.log`) |
| `test_qwen3_next_norm.py::test_cuda_parameter_contributions_reproduce_backward` (8 cases) | 8 pass (`current-pin-test.log`) | 4 fail, 4 pass (`old-implementation-pin-test.log`) |
