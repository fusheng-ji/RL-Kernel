# H100 re-run for #557 at head `7261a37a` (2026-11-12/13)

4 x NVIDIA H100 80GB (driver 580.173.02), torch 2.13.0+cu130, vLLM 0.30.0, `KERNEL_ALIGN_FORCE_SM90=1`, `TORCH_CUDA_ARCH_LIST=9.0`. Two nodes with 4 H100 each, no 8-GPU node.

| check | result |
|---|---|
| unit tests (tests/models/qwen3_next), megatron.core imported first | 652 passed, 0 failed |
| chain preflight --ep 4 | passed |
| chain layers --ep 4 | passed |
| chain l2 --ep 4 (full-model train/prefill/decode raw-bit matrix) | passed, 16 of 16 cases bit-identical (logits, logprobs, layer states, response tokens) |
| chain l2, first attempt (before the load_checkpoint fix) | failed in create_actor (TypeError, see README) |

## What was run
`tests/models/qwen3_next` (with `megatron.core` imported first, see below) and `qwen3_next_run.py --stage all --until l2 --ep 4` (preflight, layers, l2) on four H100. The check scripts and the EP4 MoE gate were not repeated; they ran at the earlier head `5294b766` (all passed; `logs/at_5294b766_*`, `logs/ep4_moe_*`, that head's failed preflight is `logs/at_5294b766_chain_preflight_failed.log`) and neither the operator-case ids nor the VIME loader call touches them.

The head `7261a37a` is a **local, unpushed** merge head: #557 plus the two fixes below (made in #543 and merged up the stack). The fixes are not yet on the fork.

## Findings
1. **Chain `preflight` failed at the earlier head `5294b766`, independent of hardware.** `qwen3_next_operator_cases.json` listed `test_cuda_gated_parameter_vjp_contributions_match_the_fold[sigmoid-dtype0]` (and three more), but the test is also parametrized with `shape` (commit `a2037197`) and collects as `...fold[shape0-silu-dtype0]`; the runner stopped with `not found`. The same stale ids were in #543, #544, #549, #551 and #557. Fixed by `62cc1225` (ids regenerated from pytest collection; four new test ids added). With it `preflight` and `layers` pass.
2. **`l2` could not run: `load_checkpoint()` was called without `skip_load_to_model_and_opt`.** The VIME loader (`rl-align-cuda` @ `6c031690`) requires it; both `create_actor` calls omitted it, so the first attempt failed with a `TypeError` before any model work (`logs/chain_ep4_to_l2_run1_failed.log`). Fixed by `1c681029`. The H100 80GB run also needs the model loaded without DDP and without an optimizer for the recompute; that is `fd87c0e1` (`create_actor(training=False)` uses `get_model(..., wrap_with_ddp=False)`). With both, `l2` passes.
3. **15 failures in `test_qwen3_next_training_checkpoint.py` without the pre-import are environmental.** The fixture patches `torch.cuda.device_count` to 0; the tests then import Megatron-LM, whose dynamic context imports `flashinfer`, which calls `torch.cuda.get_device_properties(0)` and raises `Invalid device id`. Importing `megatron.core` before pytest avoids it. CI has no Megatron-LM, so they are skipped there. No file was changed for this.

Raw logs are private; paths and host names are replaced (see `redaction.json`).
