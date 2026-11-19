# #547 at `eb465fd8`

- GPU: 4 x NVIDIA H100 per node (95830 MiB each), driver 580.173.02, CUDA 13.0 (toolkit 13.x), torch 2.13.0+cu130, triton 3.7.1, python 3.12.15, vLLM 0.30.0. Exact per-run values: `header.txt`. Topology (`nvidia-smi.txt`): the 4 GPUs of a node are pairwise connected by NVLink (NV6); no PCIe-only GPU pair.
- Build: `TORCH_CUDA_ARCH_LIST=9.0`, `MAX_JOBS=32`, `ninja` on `PATH`. Before each GPU run `rl_engine._C.__file__` was printed and is inside the worktree of the run (`c-file.txt` / `steps.txt`).
- `git status --porcelain --untracked-files=no` was empty before (`header.txt`) and after (`status-after.txt`, empty) every run. No file in a worktree was edited; no patch was applied to any run in this directory.
- The shared environment was not modified. `ninja` (and, for a diagnostic of task A only, a different `apache-tvm-ffi`) come from separate directories.
- Node: the nodes were exclusive to this work. On node 1 two runs (#495 on GPUs 0-1, #538 on GPUs 2-3) ran at the same time in separate one-pair steps; all other runs were alone on their node, and no foreign process was visible (`header.txt`).

## Commands (`steps.txt`)
Build with `KERNEL_ALIGN_FORCE_SM90=1 TORCH_CUDA_ARCH_LIST=9.0`, then, all with `KERNEL_ALIGN_FORCE_SM90=1` (all 4 GPUs of the node visible, tests use one):
- `pytest -p no:cacheprovider -q -rs tests/ops/logprob/test_deterministic_logp_recursive.py` (`pytest-recursive.log`)
- `RL_KERNEL_LOGP_MAX_ORDER=10 pytest ... test_deterministic_logp_recursive.py -k 'recursive_orders and dtype3'` (`pytest-max-order-10.log`)
- `pytest ... tests/ops/logprob tests/runtime/test_layout_compatibility.py` (`pytest-logprob-layout.log`)
- `python tools/validation/operators/probe_deterministic_logp_autograd.py --output recursive-backends.json` (`probe.log`: "Verified 48 fused/recursive/torch.func CUDA traces"; exit code 0 in `steps.txt`; output `recursive-backends.json`)

## Results
### pytest-logprob-layout.log

Result line: `640 passed, 17 skipped, 18 warnings in 110.57s (0:01:50)`

Skip reasons (`-rs`):

- 15 x batch_invariant_logp_ascend kernel not compiled (needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).
- 1 x Triton linear log-prob requires a MUSA device and Triton.
- 1 x CUDA dispatch may select fused logp

### pytest-max-order-10.log

Result line: `8 passed, 81 deselected in 16.12s`

### pytest-recursive.log

Result line: `89 passed, 18 warnings in 57.01s`


## Reading
No failure. The SM90-only tests (`batch_invariant_logp_sm90`, fused linear logp SM90) ran in this build; the 17 skips in `pytest-logprob-layout.log` are listed with their reasons above.

## Draft "H100" paragraph
4 x H100 node (driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130), head `eb465fd8` (only merges of `main` on top of the previously tested head), `KERNEL_ALIGN_FORCE_SM90=1` build, detached clean worktree, no patch: `test_deterministic_logp_recursive.py` 89 passed; the `RL_KERNEL_LOGP_MAX_ORDER=10` case 8 passed (81 deselected); `tests/ops/logprob` + `tests/runtime/test_layout_compatibility.py` 640 passed / 0 failed / 17 skipped (reasons in the log); the recursive-backends probe exits 0 and verifies 48 fused/recursive/torch.func CUDA traces.
