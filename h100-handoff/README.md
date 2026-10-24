# H100 handoff: split branches for #539 and #543

Purpose: run the Hopper-only and multi-GPU tests that could not be run on the B200 box, on the six
split branches. All six are on the fork; nothing here is a pull request yet.

Limit being enforced: every PR is at most 10,000 lines of its own diff over its parent layer.

## Branches (fork `origin`)

| layer | branch | head | parent | own diff |
|---|---|---|---|---|
| 539a | `split/h3-539a-one-block` | `cbdf48b0` | head of #494 (`feat/h3-sp-norm-adaln`) | 3.6k |
| 539b | `split/h3-539b-full-model-qualification` | `70b57205` | 539a | 8.8k |
| 539c | `split/h3-539c-generation-gate` | `44b7f832` | 539b | 1.5k |
| 543a | `split/qn-543a-foundation` | `1ee46a34` | head of #507 (`f9b85d15`) | 5.2k |
| 543b | `split/qn-543b-training-l2` | `5778bc88` | 543a | 7.5k |
| 543c | `split/qn-543c-rollout-runners` | `ac902598` | 543b | 6.6k |

The top of each stack has the same tree as the original PR head (`17d0e9e9` for #539, `4f61a39c` for #543).

## Run

```bash
REPO=<clone of the fork> WEIGHTS=<dir with h3_conditioning.safetensors and h3_block0.safetensors> \
  bash h100-handoff/run_split_tests.sh
```

If the weights are missing: `python tools/weights/prepare_h3_weights.py --out <dir> --block`.
Put `ninja` on `PATH` first. Optional: `EXTRA_PYTHONPATH=<dir>` with sglang, matplotlib and megatron-core installed there
makes `test_h3_timestep_sinusoid` (sglang bitwise parity), `test_h3_report` (plots) and the 4 megatron tests run instead of skip.
`SKIP_BUILD=1` reuses existing builds.

What it does: one worktree per branch; 539a-c with `RL_KERNEL_H3_STRICT=1` (builds `_C` and `_H3_C`) and 2 GPUs;
the NCCL 4/8-GPU byte-equality tests from 539a with all GPUs visible and `WORLD_SIZE` unset; 543a-c with a normal build,
`tests/models/qwen3_next` plus `tests/validation/common`, and the SM90-only
`test_sm90_diagnostic_path_reports_direct_lse_without_fallback` on one GPU.

## Reference results on B200 (same commits)

Plain run (2 GPUs, weights, strict build, no optional packages):

| layer | pytest | skipped, and why |
|---|---|---|
| 539a | 538 passed | 6: 4/8-GPU NCCL (2 files), matplotlib, sglang |
| 539b | 547 passed | same 6 |
| 539c | 569 passed | same 6 |
| 543a | 388 passed | 2: SM90 test needs Hopper, 4-GPU ROCm test |
| 543b | 548 passed | 6: also 4 need `megatron` |
| 543c | 640 passed | 6: same as 543b |

Skips that were then resolved with separate runs on the same commits:

- 4/8-GPU NCCL (`test_h3_sp_norm_adaln`, `test_h3_tp_adaln`): 6 passed per branch (2/4/8 GPUs) on one 8 x B200 node, at 539a and 539c.
- matplotlib and sglang 0.5.22 (installed with `pip install --no-deps --target` and put on `PYTHONPATH`; sglang's JIT needs `ninja` on `PATH`):
  `test_h3_timestep_sinusoid.py` plus `test_h3_report.py` = 49 passed on each of 539a, 539b and 539c, including the sglang bitwise-parity test.
- megatron-core 0.16.1 (same `--target` install): `test_qwen3_next_training_checkpoint.py` = 25 passed on 543b and on 543c, including the 4 that skip without it.
  This needs `megatron.core` imported before pytest starts (see `preimport_pytest.py`): the file's fixture patches
  `torch.cuda.device_count` to 0, and when megatron-core is installed next to flashinfer the first import then fails with
  "Invalid device id" (15 of 25 tests fail). `run_split_tests.sh` already goes through that wrapper.

What remains skipped on B200: the SM90 test (Hopper only) and the 4-GPU ROCm test (not NVIDIA).
543b once failed one test in a first full run (test name not captured) and passed in 6 later runs.

## Expected on H100

- The SM90 test runs and passes.
- NCCL parameters run for every GPU count the node has; 8-GPU parameters skip on a 4-GPU node.
- Still skipped: the 4-GPU ROCm test, and the `megatron`-dependent tests unless megatron-core is on `EXTRA_PYTHONPATH`.
- Paths that are Hopper-only (`det_gemm` SM90) execute here and did not on B200. A failure there is not
  necessarily caused by the split: compare against the same test on `test-h3` / `test-qwennext` before concluding.

## Report back

`results/env.txt` (GPU model, driver, torch), the last line of every `*-h3.log`, `*-qn.log`, `*-sm90.log` and
`539a-nccl-4-8gpu.log`, and the first line of each log (`_C` path, to confirm the extension is from that worktree).
For any failure: `pytest -x --tb=long -rf` output for that test and the branch head from `git rev-parse --short=8 HEAD`.
