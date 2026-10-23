# Qwen3-Next on Hopper (4 x H100 80GB HBM3): supplementary runs

Hardware/software: 4 x NVIDIA H100, driver 580, torch 2.13.0+cu130, vLLM 0.30.0, NCCL 2.29.7, checkpoint `Qwen3-Next-80B-A3B-Instruct` revision `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`. Each PR was built and run at its own head (see `info.txt` / `chain/*.json`). Build flags: `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 TORCH_CUDA_ARCH_LIST=9.0`.

Notes
- #543 and #544 gate the four-GPU preflight on B200. To run them, commit `2a8f18e4` (Hopper allowance, from #557) was applied on top of each head (`pr543/hopper_preflight_2a8f18e4.patch`).
- Scope was `l2` and below. The full-model L2 recompute (Megatron) runs out of memory on 4 x H100 (its FP32 gradient buffer, 72-74 GiB, does not fit next to the weights in 93 GiB). The vLLM rollout half of L2 completed (16 rollout artifacts).
- The environment has numpy 1.26.4 (Megatron requires numpy < 2), other runs on B200 used numpy 2.x.
- FlashInfer's CUTLASS fused-MoE JIT does not build for sm90a here; recorded as unavailable in the prior-art tables.

Directories: `pr503` (MoE route/combine), `pr507` (TP4 mixers), `pr543`/`pr544`/`pr557/chain` (stage results), `pr549` (TP4 logprob parity, 13 NCCL settings), `pr557` (EP4 gates and the 13-setting EP matrix), `pr503/diagnostics` (backward batch-invariance root cause).
