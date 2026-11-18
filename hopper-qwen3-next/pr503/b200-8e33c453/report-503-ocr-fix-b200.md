# #503 OpenCodeReview fix `8e33c453`, B200 (2026-11-18)

Head: `8e33c4535bc62263af59ca75032d404000151b03` (tree `cc54d004afddc0ab6ed7364440e90540474ec95c`), parent `e514d42f`. The commit gates the empty-input bias gradient of `_SharedLinear.backward` on `needs_input_grad[2]` and corrects the `shared_moe` docstring about host synchronisation. Earlier B200 data for `e514d42f`: `../b200-e514d42f/`.

Environment: NVIDIA B200, driver 580.126.20, CUDA 13.0, python 3.12.14, torch 2.13.0+cu130, triton 3.7.1, vllm 0.30.0, transformers 5.17.0, `VLLM_BATCH_INVARIANT=0 TRITON_F32_DEFAULT=ieee`, no SM90 flags (the versions were read from the same Python environment after the runs; the jobs did not record them). Each job used its own fresh, detached worktree at `8e33c453` and built `_C` on the B200 node with `python setup.py build_ext --inplace`; the job stops before pytest unless the realpath of `rl_engine._C` lies inside that worktree. `git status --porcelain` is empty before and after both runs.

| run | GPUs | step | result |
|---|---|---|---|
| `run-4gpu-full-suite` | 4 | `tests/models/qwen3_next/test_*.py` + `check_*.py` (9 files, same as `b200-e514d42f`) | 323 passed, 0 failed, 0 skipped, rc 0 |
| `run-4gpu-full-suite` | 4 | `bias_check.py` | rc 0 |
| `run-1gpu-forward` | 1 | `check_qwen3_next_forward.py` + `test_qwen3_next_forward_contract.py` | 36 passed, 0 failed, 0 skipped, rc 0 |
| `run-1gpu-forward` | 1 | `bias_check.py` | rc 0 |

`bias_check.py` (identical in both runs) passes an empty `(0, 2048)` bf16 input through `shared_linear` with a 64-wide bias and asserts: with `requires_grad=True` the bias gradient is a zero `(64,)` tensor; with `requires_grad=False`, `bias.grad is None`. The pass count is the same as at `e514d42f` (323), since the commit adds no test.

Files: `run-*/header.txt` (head, tree, GPUs, environment), `steps.txt` (UTC times, commands, return codes), `pytest.log`, `bias_check.log`.
