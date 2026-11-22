# Actor attribution (torch.profiler, rank 0, actor version 3)

Buckets are record_function ranges around Megatron forward entry points; their GPU time is the GPU-side span of the range (forward only; backward kernels run outside the ranges). CUPTI tracing inflates many-small-kernel paths. native = S3-profile/native, rlkernel-before = S3-profile-base (original code + profiler), rlkernel-round1 = S3-profile/rlkernel.

### recompute logp (forward_batch), rank 0, one profiled step

| bucket | native | rlkernel-before | rlkernel-round1 |
| --- | ---: | ---: | ---: |
| wall s | 55.70 | 47.58 | 10.26 |
| GPU kernel time s (sum) | 24.01 | 28.88 | 6.95 |
| `attention` GPU s (calls) | 0.92 (324) | 13.64 (324) | 0.05 (324) |
| `rope` GPU s (calls) | 18.17 (648) | 7.84 (648) | 0.50 (648) |
| `mlp` GPU s (calls) | 1.29 (324) | 1.17 (324) | 1.11 (324) |
| `lm_head` GPU s (calls) | 0.28 (9) | 0.25 (9) | 0.20 (9) |
| `det_collective.all_reduce` GPU s (calls) | - | 0.94 (648) | 0.91 (648) |
| `det_collective.all_gather` GPU s (calls) | - | 0.00 (9) | 0.00 (9) |
| `det_collective.all_gather_many` GPU s (calls) | - | - | - |
| `det_collective.reduce_scatter` GPU s (calls) | - | - | - |

Top GPU kernels, native:

| kernel | GPU s |
| --- | ---: |
| `record_param_comms` | 0.450 |
| `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 0.357 |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 0.228 |
| `nccl:_reduce_scatter_base` | 0.228 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 0.221 |
| `nccl:_all_gather_base` | 0.221 |
| `aten::mul` | 0.152 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 0.149 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 0.084 |
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 0.077 |

Top GPU kernels, rlkernel-before:

| kernel | GPU s |
| --- | ---: |
| `void (anonymous namespace)::deterministic_all_reduce_kernel<__nv_bfloat16, 4>((anonymou...` | 0.798 |
| `aten::mm` | 0.670 |
| `_DeterministicFFNFunction` | 0.493 |
| `_DeterministicTensorParallelReduce` | 0.488 |
| `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 0.364 |
| `aten::copy_` | 0.194 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 0.175 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 0.127 |
| `aten::_fused_rms_norm` | 0.122 |
| `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, flo...` | 0.122 |

Top GPU kernels, rlkernel-round1:

| kernel | GPU s |
| --- | ---: |
| `void (anonymous namespace)::deterministic_all_reduce_kernel<__nv_bfloat16, 4>((anonymou...` | 0.773 |
| `aten::mm` | 0.674 |
| `_DeterministicFFNFunction` | 0.495 |
| `_DeterministicTensorParallelReduce` | 0.429 |
| `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 0.393 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 0.149 |
| `aten::_fused_rms_norm` | 0.123 |
| `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, flo...` | 0.123 |
| `aten::add_` | 0.110 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFlo...` | 0.110 |

### actor train (train_batch), rank 0, one profiled step

| bucket | native | rlkernel-before | rlkernel-round1 |
| --- | ---: | ---: | ---: |
| wall s | 125.14 | 120.26 | 23.72 |
| GPU kernel time s (sum) | 33.83 | 47.36 | 15.65 |
| `attention` GPU s (calls) | 0.94 (324) | 19.92 (324) | 0.05 (324) |
| `rope` GPU s (calls) | 20.12 (648) | 9.68 (648) | 0.53 (648) |
| `mlp` GPU s (calls) | 1.40 (324) | 1.18 (324) | 1.16 (324) |
| `lm_head` GPU s (calls) | 0.28 (9) | 0.25 (9) | 0.20 (9) |
| `det_collective.all_reduce` GPU s (calls) | - | 2.29 (1629) | 2.23 (1629) |
| `det_collective.all_gather` GPU s (calls) | - | 0.00 (9) | 0.00 (9) |
| `det_collective.all_gather_many` GPU s (calls) | - | - | - |
| `det_collective.reduce_scatter` GPU s (calls) | - | - | - |

Top GPU kernels, native:

| kernel | GPU s |
| --- | ---: |
| `record_param_comms` | 1.157 |
| `_LayerNormLinearBackward` | 0.833 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 0.701 |
| `nccl:_all_gather_base` | 0.701 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT` | 0.519 |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 0.455 |
| `nccl:_reduce_scatter_base` | 0.455 |
| `_LinearBackward` | 0.433 |
| `_LayerNormLinear` | 0.413 |
| `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 0.357 |

Top GPU kernels, rlkernel-before:

| kernel | GPU s |
| --- | ---: |
| `aten::mm` | 2.003 |
| `void (anonymous namespace)::deterministic_all_reduce_kernel<__nv_bfloat16, 4>((anonymou...` | 1.991 |
| `_DeterministicFFNFunctionBackward` | 0.939 |
| `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFlo...` | 0.526 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT` | 0.514 |
| `aten::add_` | 0.511 |
| `_DeterministicFFNFunction` | 0.493 |
| `_DeterministicTensorParallelReduce` | 0.488 |
| `_DeterministicCopyToTensorParallelRegionBackward` | 0.487 |
| `aten::copy_` | 0.376 |

Top GPU kernels, rlkernel-round1:

| kernel | GPU s |
| --- | ---: |
| `aten::mm` | 2.018 |
| `void (anonymous namespace)::deterministic_all_reduce_kernel<__nv_bfloat16, 4>((anonymou...` | 1.944 |
| `_DeterministicFFNFunctionBackward` | 0.942 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT` | 0.518 |
| `_DeterministicFFNFunction` | 0.495 |
| `_DeterministicCopyToTensorParallelRegionBackward` | 0.430 |
| `_DeterministicTensorParallelReduce` | 0.429 |
| `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 0.394 |
| `aten::_fused_rms_norm_backward` | 0.335 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN` | 0.276 |
