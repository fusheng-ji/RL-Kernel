# #503 combine_routes route-weight gradient, H100 (2026-11-14)

Base A: #503 public head `d5253a5c` (tree `e5c0a79b`), no patch.
Base B: #503 head `d5253a5c` + one local candidate commit from `0001-fix-qwen3-next-fixed-order-route-weight-gradient-in.patch` (patch sha256 `34376300…`), `git am` HEAD `b960f8ce`, tree `ce3f499d23552b6e081f69cc070a2813a8faede2` (matches the handoff). Not a PR head.

Environment: 4 x NVIDIA H100 per run (one node each), driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, vllm 0.30.0, transformers 5.17.0, python 3.12.15, `VLLM_BATCH_INVARIANT=0`, `KERNEL_ALIGN_FORCE_SM90=1 KERNEL_ALIGN_DET_GEMM_SM90=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32`. `_C.__file__` inside each worktree. `git status --short` empty before and after in both. The probe was run from outside the worktree with `PYTHONPATH=.` so the tree stayed clean. The B node showed 0.5 to 1.9 GiB resident GPU memory at start, 0% utilization; no other job of mine was running there.

| step (UTC times in steps.txt) | A | B |
|---|---|---|
| `tests/models/qwen3_next/test_*.py` + `check_*.py` | 322 passed, 0 failed, 0 skipped, rc 0 | 323 passed, 0 failed, 0 skipped, rc 0 |
| `check_qwen3_next_forward.py -k combine_gradients` | n/a | 1 passed, rc 0 |
| TP4 MoE gate, 4 ranks | 4/4 passed, rc 0 | 4/4 passed, rc 0 |
| probe: route-weight gradient equal at T = 8, 9 / 16 ... 1024 | yes / **no at 16, 32, 64, 128, 256, 1024** | yes / **yes at every T** |
| probe: output and expert gradients | equal at every T | equal at every T |

Full probe output: `run-*/probe.log`. Commands: `run-*/header.txt` and `steps.txt`.
