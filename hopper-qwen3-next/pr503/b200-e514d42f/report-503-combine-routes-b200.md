# #503 combine_routes route-weight gradient, B200 (2026-11-14)

Base A: #503 previous head `d5253a5c` (tree `e5c0a79b`), no change.
Base B: #503 public head `e514d42f` (tree `ce3f499d23552b6e081f69cc070a2813a8faede2`, the same tree as run B of the H100 report in `../b960f8ce/`).

Environment: 4 x NVIDIA B200 (one node; 0 MiB used and 0% utilization at start, see `nvidia-smi-start-*.csv`), driver 580.126.20, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, vllm 0.30.0, transformers 5.17.0, NCCL 2.29.7, python 3.12.14, `VLLM_BATCH_INVARIANT=0 TRITON_F32_DEFAULT=ieee`, no SM90 flags. `_C` was built on the B200 node with `python setup.py build_ext --inplace` in each fresh, detached worktree; `_C.__file__` is inside that worktree and `git status --porcelain` is empty before and after. The probe ran from outside the worktree with the worktree on `PYTHONPATH`; `probe_combine.py` is byte-identical to the one in the H100 report.

TP4 checkpoint: Qwen3-Next-80B-A3B-Instruct revision `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`. The gate reads only `model.layers.0.mlp.*`, so it ran on `model.safetensors.index.json` plus `model-00001-of-00041.safetensors` (the only shard holding those tensors; sha256 checked against the Hugging Face LFS hash). It ran in a second job on the same `_C` builds; the first attempt in the main job pointed at a missing checkpoint path and is superseded (noted in `steps.txt`).

| step (UTC times in steps.txt) | A `d5253a5c` | B `e514d42f` |
|---|---|---|
| `tests/models/qwen3_next/test_*.py` + `check_*.py` (9 files) | 322 passed, 0 failed, 0 skipped, rc 0 | 323 passed, 0 failed, 0 skipped, rc 0 |
| `check_qwen3_next_forward.py -k combine_gradients` | n/a | 1 passed, rc 0 |
| TP4 MoE gate, 4 ranks | 12/12 cases on 4/4 ranks, rc 0 | 12/12 cases on 4/4 ranks, rc 0 |
| probe: route-weight gradient equal at T = 8, 9 / 16 ... 1024 | yes / **no at 16, 32, 64, 128, 256, 1024** | yes / **yes at every T** |
| probe: output and expert gradients | equal at every T | equal at every T |

B200 reproduces the H100 result: the stock `torch.sum` backward breaks route-weight batch invariance from 16 rows on both architectures, and `e514d42f` fixes it on both. Pass counts match the H100 run (322 / 323).

Full probe output: `run-*/probe.log`. Commands and environment: `run-*/header.txt` and `steps.txt`.
