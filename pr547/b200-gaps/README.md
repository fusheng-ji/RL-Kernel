# PR #547 on B200: repo-reproducible commands at the PR head

Head `ff1ad6350050b3b11bec86364f5c1bfaecf23f78`, clean tree, extension built `TORCH_CUDA_ARCH_LIST=10.0`, one B200, driver 580.126.20, CUDA 13.0, **torch 2.13.0+cu130** (the PR's recorded validation used PyTorch 2.9.1). Runner: [run.sh](run.sh). The rest of the PR's acceptance (`run-validation.sh`) lives in a validation workspace and was not rerun.

| Run | passed | failed | skipped |
|---|---:|---:|---:|
| `test_deterministic_logp_recursive.py` | 89 | 0 | 0 |
| order 10 (`RL_KERNEL_LOGP_MAX_ORDER=10`, `recursive_orders and dtype3`) | 8 | 0 | 81 deselected |
| all of `tests/ops/logprob` | 588 | 0 | 53 (SM90 kernels not compiled, Ascend, two-GPU TP tests with one GPU, one CUDA-dispatch guard) |
| `tests/runtime/test_layout_compatibility.py` | 16 | 0 | 0 |
| `probe_deterministic_logp_autograd.py` | 48 fused/recursive/torch.func traces verified | | |

The PR's scope note says nested JVP uses PyTorch internal forward-AD hooks and is validated on 2.9.1 only; these results show the recursive suite (including nested JVP cases) passing on torch 2.13.

Draft addition for the PR body Results:

> **Re-run with the shared torch 2.13.0+cu130 environment at head `ff1ad635` (B200).** `test_deterministic_logp_recursive.py` 89 passed; order 9 to 10 FP64 cases 8 passed; `tests/ops/logprob` 588 passed, 53 skipped (SM90 / Ascend / two-GPU); layout test 16 passed; the backend probe verified 48 traces. So the recursive and nested-JVP paths also pass on torch 2.13, not only on 2.9.1. Raw data: `pr_assets` `pr547/b200-gaps/` @ `<EVIDENCE_SHA>`.
